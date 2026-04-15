"""
Pre-segment all raw audio into fixed-length windows and save to processed/.

This script reads master_manifest.csv, loads each raw audio file, normalizes
loudness, resamples, and segments it into 4-second windows (2-second hop for
50% overlap). Output .wav files go to:

    <processed_root>/<domain>/<real_or_fake>/<source_dataset>/

A NEW manifest is written to data/metadata/master_manifest_segmented.csv
where each row is one 4-second segment (not one source file).

Running on Colab (recommended — avoids filling local C: drive):
--------------------------------------------------------------
    from google.colab import drive
    drive.mount('/content/drive')

    !git clone https://github.com/YOUR_ORG/AI-Innovation /content/AI-Innovation
    %cd /content/AI-Innovation
    !pip install -q soundfile torchaudio pyloudnorm

    DATA_ROOT   = '/content/drive/MyDrive/AI-Innovation-Data'
    PROC_ROOT   = DATA_ROOT + '/processed'
    MANIFEST_IN = 'data/metadata/master_manifest.csv'

    !python scripts/preprocess_segments.py \\
        --manifest {MANIFEST_IN} \\
        --processed-root {PROC_ROOT} \\
        --workers 4

Running locally (slow on Drive Streaming):
------------------------------------------
    python scripts/preprocess_segments.py \\
        --manifest data/metadata/master_manifest.csv \\
        --processed-root "C:/Users/fooja/Google Drive Streaming/My Drive/AI-Innovation-Data/processed" \\
        --workers 2

Performance notes:
  - ~143K source files, average 5–10 seconds each → ~300–600K segments
  - Expected runtime on Colab A100: ~3–6 hours
  - Expected runtime on local CPU: ~12–24 hours
  - Re-runnable: skips files that already exist in processed/
"""

import argparse
import csv
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Audio constants ───────────────────────────────────────────────────────────
WINDOW_SEC   = 4.0    # segment length in seconds
HOP_SEC      = 2.0    # hop between windows (50% overlap)
MIN_SEG_SEC  = 1.0    # discard segments shorter than this
TARGET_LUFS  = -23.0  # EBU R128 target loudness

SAMPLE_RATES = {
    "voice":     16_000,
    "music":     44_100,
    "non_human": 44_100,
}
DEFAULT_SR = 16_000


# ── Worker function (runs in subprocess) ─────────────────────────────────────

def _process_row(args):
    """Load one source file, segment it, return list of output rows."""
    row, processed_root, dry_run = args

    import math
    import torch
    import torchaudio
    import torchaudio.functional as AF

    src_path = Path(row["file_path"])
    if not src_path.exists():
        return [], f"MISSING: {src_path}"

    domain  = row.get("domain", "voice")
    label   = float(row.get("label", 0.0))
    real_or_fake = "fake" if label >= 0.5 else "real"
    source   = row.get("source_dataset", "unknown")
    split    = row.get("split", "train")
    generator = row.get("generator", "unknown")

    target_sr = SAMPLE_RATES.get(domain, DEFAULT_SR)
    win_samples = int(WINDOW_SEC * target_sr)
    hop_samples = int(HOP_SEC   * target_sr)
    min_samples = int(MIN_SEG_SEC * target_sr)

    # Output directory
    out_dir = Path(processed_root) / domain / real_or_fake / source
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    try:
        wav, sr = torchaudio.load(str(src_path))
    except Exception as e:
        return [], f"LOAD_ERROR ({e}): {src_path}"

    # Mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # Resample
    if sr != target_sr:
        wav = AF.resample(wav, sr, target_sr)

    # Loudness normalize (simple peak normalize as pyloudnorm may not be installed)
    peak = wav.abs().max()
    if peak > 0:
        # Convert -23 LUFS ≈ peak-normalize then scale to ~0.1 RMS
        # Full EBU R128 requires pyloudnorm; skip if unavailable
        try:
            import pyloudnorm as pyln
            import numpy as np
            meter = pyln.Meter(target_sr)
            wav_np = wav.squeeze(0).numpy().astype("float64")
            loudness = meter.integrated_loudness(wav_np)
            if loudness != float("-inf"):
                gain_db = TARGET_LUFS - loudness
                gain_linear = 10 ** (gain_db / 20)
                wav = wav * gain_linear
                # Clip to prevent overflow
                wav = wav.clamp(-1.0, 1.0)
        except ImportError:
            # Fallback: peak normalize
            wav = wav / (peak + 1e-8)

    total_samples = wav.shape[1]
    stem = src_path.stem

    output_rows = []
    seg_idx = 0
    start = 0
    while start < total_samples:
        end = start + win_samples
        chunk = wav[:, start:end]

        if chunk.shape[1] < min_samples:
            break  # too short, discard

        if chunk.shape[1] < win_samples:
            # Zero-pad the last partial segment
            pad = win_samples - chunk.shape[1]
            chunk = torch.nn.functional.pad(chunk, (0, pad))

        seg_name = f"{stem}_seg{seg_idx:04d}.wav"
        out_path = out_dir / seg_name
        sample_id = f"{source}_{stem}_seg{seg_idx:04d}"

        if not dry_run and not out_path.exists():
            torchaudio.save(str(out_path), chunk, target_sr)

        output_rows.append({
            "sample_id":      sample_id,
            "file_path":      str(out_path).replace("\\", "/"),
            "label":          label,
            "domain":         domain,
            "source_dataset": source,
            "generator":      generator,
            "split":          split,
            "seg_index":      seg_idx,
            "source_file":    str(src_path).replace("\\", "/"),
        })

        seg_idx += 1
        start += hop_samples

    return output_rows, None


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pre-segment audio for training")
    parser.add_argument("--manifest",        default="data/metadata/master_manifest.csv")
    parser.add_argument("--data-root",       default=None,
                        help="Root dir prepended to relative file_path entries in the manifest. "
                             "Required when manifest uses relative paths. "
                             "E.g.: 'C:/Users/you/Google Drive Streaming/My Drive/AI-Innovation-Data'")
    parser.add_argument("--processed-root",  required=True,
                        help="Root dir for output .wav segments")
    parser.add_argument("--out-manifest",    default="data/metadata/master_manifest_segmented.csv")
    parser.add_argument("--workers",         type=int, default=4)
    parser.add_argument("--dry-run",         action="store_true",
                        help="Count segments without writing files")
    parser.add_argument("--split",           default=None,
                        help="Process only this split (train|val|test)")
    parser.add_argument("--source",          default=None,
                        help="Process only this source_dataset (e.g. ljspeech)")
    parser.add_argument("--limit",           type=int, default=None,
                        help="Process at most N source files (for smoke test)")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        log.error("Manifest not found: %s", manifest_path)
        sys.exit(1)

    with open(manifest_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Prepend data_root to relative paths so workers can find files on disk
    if args.data_root:
        data_root = Path(args.data_root)
        for row in rows:
            fp = row["file_path"]
            if not Path(fp).is_absolute():
                row["file_path"] = str(data_root / fp)

    # Optional filters
    if args.split:
        rows = [r for r in rows if r["split"] == args.split]
    if args.source:
        rows = [r for r in rows if r["source_dataset"] == args.source]
    if args.limit:
        rows = rows[: args.limit]

    log.info("Processing %d source files  (workers=%d  dry_run=%s)",
             len(rows), args.workers, args.dry_run)

    work = [(row, args.processed_root, args.dry_run) for row in rows]
    all_seg_rows = []
    errors = []
    done = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_row, w): w for w in work}
        for fut in as_completed(futures):
            seg_rows, err = fut.result()
            all_seg_rows.extend(seg_rows)
            if err:
                errors.append(err)
            done += 1
            if done % 1000 == 0:
                log.info("  %d / %d files done  (%d segments so far)",
                         done, len(rows), len(all_seg_rows))

    log.info("Done: %d source files → %d segments  (%d errors)",
             len(rows), len(all_seg_rows), len(errors))

    if errors[:10]:
        log.warning("First errors:\n  %s", "\n  ".join(errors[:10]))

    if not args.dry_run and all_seg_rows:
        out_path = Path(args.out_manifest)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(all_seg_rows[0].keys())
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_seg_rows)
        log.info("Segmented manifest written: %s  (%d rows)", out_path, len(all_seg_rows))

    # Summary by split/domain
    from collections import Counter, defaultdict
    by_split  = Counter(r["split"]  for r in all_seg_rows)
    by_domain = Counter(r["domain"] for r in all_seg_rows)
    by_src    = Counter(r["source_dataset"] for r in all_seg_rows)

    print("\n=== Segment counts by split ===")
    for k, v in sorted(by_split.items()):
        print(f"  {k:<8} {v:>8}")
    print("\n=== Segment counts by domain ===")
    for k, v in sorted(by_domain.items()):
        print(f"  {k:<15} {v:>8}")
    print("\n=== Segment counts by source ===")
    for k, v in sorted(by_src.items(), key=lambda x: -x[1]):
        print(f"  {k:<22} {v:>8}")


if __name__ == "__main__":
    main()

"""
Download real music clips from the MusicCaps dataset (Google Research) and
register them in the project's master manifest.

Usage:
    python scripts/fetch_musiccaps.py                  # 1000 clips, 4 workers
    python scripts/fetch_musiccaps.py --max-clips 0    # all ~5K clips
    python scripts/fetch_musiccaps.py --balanced-only  # balanced subset only
    python scripts/fetch_musiccaps.py --max-clips 200 --workers 8
"""

import argparse
import csv
import logging
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DRIVE_ROOT = Path("C:/Users/fooja/Google Drive Streaming/My Drive/AI-Innovation-Data")
OUT_DIR = DRIVE_ROOT / "raw" / "music" / "real" / "musiccaps"
MANIFEST = ROOT / "data" / "metadata" / "master_manifest.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ── Manifest helpers (same pattern as fetch_dataset.py) ──────────────────────

def _load_manifest():
    if not MANIFEST.exists():
        return [], set()
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows, {r["sample_id"] for r in rows}


def _save_manifest(rows):
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sample_id", "file_path", "label", "domain", "source_dataset", "generator", "split"]
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ── yt-dlp check ─────────────────────────────────────────────────────────────

def _check_ytdlp():
    if shutil.which("yt-dlp") is None:
        print(
            "ERROR: yt-dlp not found in PATH.\n"
            "Install it with:  pip install yt-dlp\n"
            "Then re-run this script."
        )
        sys.exit(1)


# ── Single-clip download ──────────────────────────────────────────────────────

def _download_clip(ytid, start_s, end_s, out_dir):
    output_path = out_dir / f"{ytid}.wav"

    if output_path.exists():
        return ytid, True, "already exists"

    url = f"https://www.youtube.com/watch?v={ytid}"
    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", "wav",
        "--download-sections", f"*{start_s}-{end_s}",
        "--force-keyframes-at-cuts",
        "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1",
        "-o", str(output_path),
        url,
    ]

    try:
        result = subprocess.run(
            cmd,
            timeout=60,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0 and output_path.exists():
            return ytid, True, "OK"
        else:
            stderr_tail = result.stderr.strip().splitlines()
            reason = stderr_tail[-1] if stderr_tail else f"exit code {result.returncode}"
            return ytid, False, reason
    except subprocess.TimeoutExpired:
        return ytid, False, "timeout after 60s"
    except Exception as exc:
        return ytid, False, str(exc)


# ── Main fetch function ───────────────────────────────────────────────────────

def fetch_musiccaps(max_clips=1000, workers=4, balanced_only=False):
    _check_ytdlp()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading MusicCaps metadata from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset("google/MusicCaps", split="train")

    clips = list(ds)
    if balanced_only:
        clips = [c for c in clips if c.get("is_balanced_subset")]
        log.info(f"Balanced subset: {len(clips)} rows")
    else:
        log.info(f"Full dataset: {len(clips)} rows")

    if max_clips and max_clips > 0:
        clips = clips[:max_clips]
        log.info(f"Limiting to {len(clips)} clips (--max-clips {max_clips})")

    total = len(clips)
    done_count = 0
    ok_count = 0
    fail_count = 0
    skip_count = 0

    log.info(f"Starting downloads: {total} clips, {workers} workers -> {OUT_DIR}")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _download_clip,
                c["ytid"],
                c["start_s"],
                c["end_s"],
                OUT_DIR,
            ): c["ytid"]
            for c in clips
        }

        for future in as_completed(futures):
            ytid, success, msg = future.result()
            done_count += 1

            if msg == "already exists":
                skip_count += 1
                status_tag = "SKIP"
            elif success:
                ok_count += 1
                status_tag = "OK"
            else:
                fail_count += 1
                status_tag = "FAIL"
                log.warning(f"  {ytid}: {msg}")

            print(f"[{done_count}/{total}] {ytid} {status_tag}", flush=True)

    print()
    print(f"=== Download summary ===")
    print(f"  Total   : {total}")
    print(f"  OK      : {ok_count}")
    print(f"  Skipped : {skip_count}  (already on disk)")
    print(f"  Failed  : {fail_count}")
    print()

    # ── Register in manifest ──────────────────────────────────────────────────

    log.info("Scanning output directory and updating manifest...")
    existing_rows, existing_ids = _load_manifest()

    new_rows = []
    for wav in sorted(OUT_DIR.glob("*.wav")):
        ytid = wav.stem
        sample_id = f"musiccaps_{ytid}"
        if sample_id in existing_ids:
            continue
        new_rows.append({
            "sample_id":      sample_id,
            "file_path":      str(wav).replace("\\", "/"),
            "label":          0.0,
            "domain":         "music",
            "source_dataset": "musiccaps",
            "generator":      "human",
            "split":          "train",
        })

    if not new_rows:
        log.info("Manifest already up to date. Nothing new to register.")
        return

    _save_manifest(existing_rows + new_rows)
    log.info(f"Manifest updated: registered {len(new_rows)} new MusicCaps files.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download MusicCaps clips and register them in the master manifest."
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=1000,
        metavar="N",
        help="Max clips to download (default: 1000; 0 = all ~5K).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        metavar="N",
        help="Parallel download workers (default: 4).",
    )
    parser.add_argument(
        "--balanced-only",
        action="store_true",
        help="Only download rows in the balanced subset (~5K).",
    )
    args = parser.parse_args()
    fetch_musiccaps(
        max_clips=args.max_clips,
        workers=args.workers,
        balanced_only=args.balanced_only,
    )

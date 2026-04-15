"""
Build master_manifest_segmented.csv by scanning the processed/ directory.

Handles two folder structures that exist in processed/:

  NEW (from preprocess_segments.py):
    processed/<domain>/<real_or_fake>/<source>/<stem>_seg<N>.wav
    e.g. processed/voice/real/asvspoof2019/LA_T_xxxx_seg0000.wav
         processed/music/real/musiccaps/abcde_seg0001.wav
         processed/non_human/fake/audiogen/audiogen_0001_seg0000.wav

  OLD (from Camille's earlier preprocessing run):
    processed/Voice/<subfolder>/<filename>.wav
    e.g. processed/Voice/real/ljspeech/LJ001-0001_seg0000.wav
         processed/Voice/Anh-ASVspoof 2021 voice clips/AI voice/001a051b0738.wav

Labels, domains, and sources are inferred from path structure.
Splits are recovered from the original manifest where possible,
otherwise assigned deterministically by MD5 hash of the filename stem.

Usage:
    python scripts/build_segmented_manifest.py \\
        --processed-root "C:/Users/you/.../AI-Innovation-Data/processed" \\
        --manifest data/metadata/master_manifest.csv \\
        --out data/metadata/master_manifest_segmented.csv
"""

import argparse
import csv
import hashlib
import re
from collections import Counter
from pathlib import Path

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15


def assign_split(stem: str) -> str:
    h = int(hashlib.md5(stem.encode()).hexdigest(), 16) % 100
    if h < int(TRAIN_RATIO * 100):
        return "train"
    elif h < int((TRAIN_RATIO + VAL_RATIO) * 100):
        return "val"
    return "test"


def load_original_manifest(manifest_path: str) -> dict:
    """Build a lookup: original_stem -> {split, label, domain, source, generator}"""
    lookup = {}
    if not Path(manifest_path).exists():
        return lookup
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fp = Path(row["file_path"])
            stem = fp.stem  # e.g. "LA_T_1138215" or "LJ001-0001"
            lookup[stem] = {
                "split":          row["split"],
                "label":          float(row["label"]),
                "domain":         row["domain"],
                "source_dataset": row["source_dataset"],
                "generator":      row["generator"],
            }
    return lookup


def parse_segment_file(wav_path: Path, processed_root: Path, orig_lookup: dict) -> dict | None:
    """
    Parse one .wav file in processed/ and return a manifest row dict, or None to skip.
    """
    rel = wav_path.relative_to(processed_root)
    parts = rel.parts   # e.g. ('voice', 'real', 'asvspoof2019', 'LA_T_xxxx_seg0000.wav')
    stem = wav_path.stem

    # ── NEW structure: domain / real_or_fake / source / file ─────────────────
    # parts[1] MUST be "real" or "fake" to distinguish from old Voice/ structure
    if (len(parts) >= 4
            and parts[0].lower() in ("voice", "music", "non_human")
            and parts[1].lower() in ("real", "fake")):
        domain      = parts[0].lower()
        real_or_fake = parts[1].lower()
        source      = parts[2]
        label       = 1.0 if real_or_fake == "fake" else 0.0

        # Try to look up split from original manifest by stripping _segNNNN
        orig_stem = re.sub(r"_seg\d+$", "", stem)
        if orig_stem in orig_lookup:
            info = orig_lookup[orig_stem]
            split     = info["split"]
            generator = info["generator"]
        else:
            split     = assign_split(stem)
            generator = source if label == 1.0 else "human"

        return {
            "sample_id":      f"{source}_{stem}",
            "file_path":      str(wav_path).replace("\\", "/"),
            "label":          label,
            "domain":         domain,
            "source_dataset": source,
            "generator":      generator,
            "split":          split,
        }

    # ── OLD structure: Voice / subfolder / file ───────────────────────────────
    # e.g. Voice/real/ljspeech/LJ001-0001_seg0000.wav
    #      Voice/Anh-ASVspoof 2021 voice clips/AI voice/001a051b0738.wav
    if len(parts) >= 2 and parts[0] == "Voice":
        domain = "voice"

        # Case 1: Voice/real/<source>/<stem>_seg<N>.wav  or  Voice/fake/<source>/...
        if parts[1].lower() in ("real", "fake") and len(parts) >= 4:
            real_or_fake = parts[1].lower()
            source       = parts[2]
            label        = 1.0 if real_or_fake == "fake" else 0.0
            orig_stem    = re.sub(r"_seg\d+$", "", stem)
            if orig_stem in orig_lookup:
                info      = orig_lookup[orig_stem]
                split     = info["split"]
                generator = info["generator"]
            else:
                split     = assign_split(stem)
                generator = source if label == 1.0 else "human"
            return {
                "sample_id":      f"{source}_{stem}",
                "file_path":      str(wav_path).replace("\\", "/"),
                "label":          label,
                "domain":         domain,
                "source_dataset": source,
                "generator":      generator,
                "split":          split,
            }

        # Case 2: Voice/Anh-ASVspoof 2021 voice clips/AI voice/<file>.wav
        folder_name = parts[1]  # "Anh-ASVspoof 2021 voice clips"
        if "asvspoof" in folder_name.lower() or "asv" in folder_name.lower():
            source = "asvspoof2021"
            # Infer label from subfolder name
            subfolder = parts[2].lower() if len(parts) >= 3 else ""
            if "ai" in subfolder or "fake" in subfolder or "spoof" in subfolder:
                label = 1.0
                generator = "unknown"
            else:
                label = 1.0  # default: treat all as fake unless explicitly real
                generator = "unknown"
            # "Human voice" folder -> real
            if "human" in subfolder or "real" in subfolder or "bonafide" in subfolder:
                label = 0.0
                generator = "human"

            # Try manifest lookup first (by stem)
            orig_stem = re.sub(r"_seg\d+$", "", stem)
            if orig_stem in orig_lookup:
                info      = orig_lookup[orig_stem]
                split     = info["split"]
                label     = info["label"]
                generator = info["generator"]
            else:
                split = assign_split(stem)

            return {
                "sample_id":      f"{source}_{stem}",
                "file_path":      str(wav_path).replace("\\", "/"),
                "label":          label,
                "domain":         domain,
                "source_dataset": source,
                "generator":      generator,
                "split":          split,
            }

    # Unknown structure — skip
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", required=True,
                        help="Root of the processed/ directory")
    parser.add_argument("--manifest",       default="data/metadata/master_manifest.csv",
                        help="Original raw manifest (for split/label lookup)")
    parser.add_argument("--out",            default="data/metadata/master_manifest_segmented.csv")
    args = parser.parse_args()

    processed_root = Path(args.processed_root)
    if not processed_root.exists():
        print(f"ERROR: processed-root not found: {processed_root}")
        return

    print(f"Loading original manifest for split lookup: {args.manifest}")
    orig_lookup = load_original_manifest(args.manifest)
    print(f"  -> {len(orig_lookup)} entries loaded")

    print(f"\nScanning {processed_root} ...")
    rows = []
    skipped = 0
    for wav in sorted(processed_root.rglob("*.wav")):
        row = parse_segment_file(wav, processed_root, orig_lookup)
        if row:
            rows.append(row)
        else:
            skipped += 1

    print(f"Found {len(rows)} segments ({skipped} skipped/unrecognised)")

    # Stats
    src_counts  = Counter(r["source_dataset"] for r in rows)
    split_counts = Counter(r["split"] for r in rows)
    label_counts = Counter("real" if float(r["label"]) == 0.0 else "fake" for r in rows)

    print("\n=== By source ===")
    for k, v in sorted(src_counts.items(), key=lambda x: -x[1]):
        print(f"  {k:<25} {v:>8}")
    print("\n=== By split ===")
    for k, v in sorted(split_counts.items()):
        print(f"  {k:<10} {v:>8}")
    print("\n=== By label ===")
    for k, v in label_counts.items():
        print(f"  {k:<10} {v:>8}")

    # Write
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sample_id", "file_path", "label", "domain",
                  "source_dataset", "generator", "split"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()

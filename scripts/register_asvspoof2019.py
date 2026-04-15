"""
Register ASVspoof 2019 LA dataset into master_manifest.csv.

Reads the official protocol files (which define labels AND splits),
maps filenames to actual .flac paths, and appends to the manifest.

Expected directory structure after downloading and extracting:
    <asvspoof_root>/
    ├── ASVspoof2019_LA_cm_protocols/
    │   ├── ASVspoof2019.LA.cm.train.trn.txt
    │   ├── ASVspoof2019.LA.cm.dev.trl.txt
    │   └── ASVspoof2019.LA.cm.eval.trl.txt
    ├── ASVspoof2019_LA_train/
    │   └── flac/
    ├── ASVspoof2019_LA_dev/
    │   └── flac/
    └── ASVspoof2019_LA_eval/
        └── flac/

Usage:
    python scripts/register_asvspoof2019.py --root "C:/path/to/ASVspoof2019_LA"

Split mapping (follows official ASVspoof benchmark convention):
    train  -> manifest split=train
    dev    -> manifest split=val
    eval   -> manifest split=test
"""

import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MANIFEST = ROOT / "data" / "metadata" / "master_manifest.csv"

# Protocol filename → (manifest split, flac subdir name)
SPLITS = {
    "ASVspoof2019.LA.cm.train.trn.txt": ("train", "ASVspoof2019_LA_train"),
    "ASVspoof2019.LA.cm.dev.trl.txt":   ("val",   "ASVspoof2019_LA_dev"),
    "ASVspoof2019.LA.cm.eval.trl.txt":  ("test",  "ASVspoof2019_LA_eval"),
}


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


def _parse_protocol(proto_path):
    """Parse an ASVspoof 2019 LA protocol file.

    Protocol format (space-separated, 5 columns):
        SPEAKER_ID  FILENAME  -  SYSTEM_ID  LABEL
        LA_0079     LA_T_1138215  -  -       bonafide
        LA_0051     LA_T_1271820  -  A01     spoof

    Returns list of (filename, system_id, label_int) tuples.
    """
    entries = []
    with open(proto_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            filename  = parts[1]   # e.g. LA_T_1138215
            system_id = parts[3]   # e.g. A01 or - (bonafide)
            label_str = parts[4]   # bonafide | spoof
            label = 0 if label_str == "bonafide" else 1
            entries.append((filename, system_id, label))
    return entries


def main(asvspoof_root: str, dry_run: bool = False):
    root = Path(asvspoof_root)
    proto_dir = root / "ASVspoof2019_LA_cm_protocols"

    if not proto_dir.exists():
        # Some extracted archives drop the top-level dir
        proto_dir = root
        if not any(proto_dir.glob("ASVspoof2019.LA.cm.*.txt")):
            print(f"ERROR: Could not find protocol files under {root}")
            print("Expected: ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt")
            sys.exit(1)

    existing_rows, existing_ids = _load_manifest()
    print(f"Existing manifest rows: {len(existing_rows)}")

    new_rows = []
    stats = {}

    for proto_name, (split, flac_subdir) in SPLITS.items():
        proto_path = proto_dir / proto_name
        if not proto_path.exists():
            print(f"WARNING: Protocol file not found: {proto_path} — skipping {split}")
            continue

        flac_dir = root / flac_subdir / "flac"
        if not flac_dir.exists():
            # Try without nested flac/ subdir
            flac_dir = root / flac_subdir
            if not flac_dir.exists():
                print(f"WARNING: Audio dir not found: {root / flac_subdir} — skipping {split}")
                continue

        entries = _parse_protocol(proto_path)
        split_new = split_skip = split_missing = 0

        # List directory once into a set — avoids 25K+ individual path.exists() calls
        # which are very slow on network-backed drives (e.g. Google Drive Streaming).
        print(f"  Scanning {flac_dir} ...")
        existing_files = {f.lower() for f in os.listdir(flac_dir)} if flac_dir.exists() else set()
        print(f"  Found {len(existing_files)} files on disk for split={split}")

        for filename, system_id, label in entries:
            sample_id = f"asvspoof2019_{filename}"
            if sample_id in existing_ids:
                split_skip += 1
                continue

            # Check membership in pre-built set (O(1), no network I/O per file)
            flac_name = f"{filename}.flac"
            wav_name  = f"{filename}.wav"
            if flac_name.lower() in existing_files:
                flac_path = flac_dir / flac_name
            elif wav_name.lower() in existing_files:
                flac_path = flac_dir / wav_name
            else:
                split_missing += 1
                continue

            generator = "human" if label == 0 else (system_id if system_id != "-" else "unknown")

            new_rows.append({
                "sample_id":      sample_id,
                "file_path":      str(flac_path).replace("\\", "/"),
                "label":          float(label),
                "domain":         "voice",
                "source_dataset": "asvspoof2019",
                "generator":      generator,
                "split":          split,
            })
            existing_ids.add(sample_id)
            split_new += 1

        stats[split] = (split_new, split_skip, split_missing)
        print(f"  {split:5s}: {split_new:6d} new | {split_skip:6d} skipped | {split_missing:6d} missing files")

    print(f"\nTotal new rows: {len(new_rows)}")

    if not new_rows:
        print("Nothing to add.")
        return

    if dry_run:
        print("Dry run — manifest not written.")
        return

    all_rows = existing_rows + new_rows
    _save_manifest(all_rows)
    print(f"Manifest updated: {len(all_rows)} total rows -> {MANIFEST}")

    # Summary
    from collections import Counter
    label_counts  = Counter(r["label"]          for r in all_rows)
    source_counts = Counter(r["source_dataset"] for r in all_rows)
    split_counts  = Counter(r["split"]          for r in all_rows)

    print("\n=== Manifest Summary ===")
    print(f"  Total  : {len(all_rows)}")
    print(f"  Real   : {label_counts.get('0.0', 0)}")
    print(f"  AI     : {label_counts.get('1.0', 0)}")
    print(f"  Splits : {dict(split_counts)}")
    print(f"  Sources: {dict(source_counts)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register ASVspoof 2019 LA into master_manifest.csv")
    parser.add_argument(
        "--root",
        required=True,
        help='Path to ASVspoof2019_LA root dir, e.g. "C:/Users/fooja/Google Drive Streaming/My Drive/AI-Innovation-Data/raw/voice/ASVspoof2019_LA"',
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and count rows without writing to manifest",
    )
    args = parser.parse_args()
    main(args.root, dry_run=args.dry_run)

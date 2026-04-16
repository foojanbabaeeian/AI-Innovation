"""
Append generated AI audio clips into master_manifest.csv as new FAKE rows.

After running generate_musicgen.py or generate_audiogen.py on Colab and syncing
the output .wav files to the Drive raw/ tree, run this script LOCALLY (or on
Colab) to register the new files in master_manifest.csv.

Splits are assigned via MD5(sample_id) so they're stratified and reproducible,
matching the hashing used in balance_manifest.py / rebalance_splits.py.

Usage:
    # After MusicGen:
    python scripts/register_generated.py \\
        --dir data/raw/music/fake/musicgen \\
        --domain music --source musicgen --generator musicgen

    # After AudioGen:
    python scripts/register_generated.py \\
        --dir data/raw/non_human/fake/audiogen_v2 \\
        --domain non_human --source audiogen_v2 --generator audiogen_v2

Options:
    --data-root    Absolute root to make file_path relative (default: project root).
    --dry-run      Show what would be added without writing.
"""

import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "metadata" / "master_manifest.csv"

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15


def assign_split(sample_id: str) -> str:
    h = int(hashlib.md5(sample_id.encode()).hexdigest(), 16) % 100
    if h < int(TRAIN_RATIO * 100):
        return "train"
    if h < int((TRAIN_RATIO + VAL_RATIO) * 100):
        return "val"
    return "test"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir",       required=True, help="Directory of generated .wav files")
    p.add_argument("--domain",    required=True, choices=["voice", "music", "non_human"])
    p.add_argument("--source",    required=True, help="source_dataset column value")
    p.add_argument("--generator", required=True, help="generator column value")
    p.add_argument("--data-root", default=str(ROOT),
                   help="Root to make file_path relative against (default: project root)")
    p.add_argument("--label",     type=float, default=1.0, help="0.0 real, 1.0 fake (default 1.0)")
    p.add_argument("--dry-run",   action="store_true")
    args = p.parse_args()

    gen_dir = Path(args.dir).resolve()
    if not gen_dir.exists():
        print(f"ERROR: directory does not exist: {gen_dir}")
        return

    data_root = Path(args.data_root).resolve()

    # Load existing manifest (if any) so we can skip duplicates.
    existing_ids: set[str] = set()
    existing_rows: list[dict] = []
    if MANIFEST.exists():
        with open(MANIFEST, newline="", encoding="utf-8") as f:
            existing_rows = list(csv.DictReader(f))
        existing_ids = {r["sample_id"] for r in existing_rows}

    # Build new rows.
    new_rows = []
    for wav in sorted(gen_dir.glob("*.wav")):
        sample_id = f"{args.source}_{wav.stem}"
        if sample_id in existing_ids:
            continue

        try:
            file_path = str(wav.relative_to(data_root)).replace("\\", "/")
        except ValueError:
            # Not under data_root — fall back to absolute path.
            file_path = str(wav).replace("\\", "/")

        new_rows.append({
            "sample_id":      sample_id,
            "file_path":      file_path,
            "label":          args.label,
            "domain":         args.domain,
            "source_dataset": args.source,
            "generator":      args.generator,
            "split":          assign_split(sample_id),
        })

    print(f"Scanning {gen_dir}")
    print(f"  Files on disk: {len(list(gen_dir.glob('*.wav')))}")
    print(f"  New rows to add: {len(new_rows)}  (duplicates skipped: {len(list(gen_dir.glob('*.wav'))) - len(new_rows)})")

    if new_rows:
        print("\n  Split distribution:")
        for k, v in Counter(r["split"] for r in new_rows).most_common():
            print(f"    {k}: {v}")

    if args.dry_run or not new_rows:
        if args.dry_run:
            print("\nDry run — manifest NOT written.")
        return

    # Write combined manifest.
    all_rows = existing_rows + new_rows
    fieldnames = ["sample_id", "file_path", "label", "domain",
                  "source_dataset", "generator", "split"]
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nManifest updated: {MANIFEST}")
    print(f"Rows: {len(existing_rows)} -> {len(all_rows)}  (+{len(new_rows)})")


if __name__ == "__main__":
    main()

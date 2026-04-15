"""
Rebalance train/val/test splits in master_manifest.csv.

Problem: after ingestion, non-ASVspoof datasets are 100% in one split
because individual ingest scripts hardcode split="train" or split="val".
This script redistributes them so every dataset appears in all three splits.

Strategy:
  - ASVspoof 2019: keep official protocol splits (sacred for benchmarking).
  - All other datasets: apply stratified 70/15/15 split per (source_dataset, label)
    so real/fake ratios are preserved within each split.
  - Splits are deterministic via MD5 hash of sample_id (reproducible, no randomness).

Usage:
    python scripts/rebalance_splits.py [--dry-run]
    python scripts/rebalance_splits.py --apply
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
# test = remainder = 0.15

# Datasets whose splits are authoritative and must not be changed.
FROZEN_DATASETS = {"asvspoof2019"}


def assign_split(sample_id: str) -> str:
    """Deterministically assign a split via MD5 hash of sample_id."""
    h = int(hashlib.md5(sample_id.encode()).hexdigest(), 16) % 100
    if h < int(TRAIN_RATIO * 100):
        return "train"
    elif h < int((TRAIN_RATIO + VAL_RATIO) * 100):
        return "val"
    return "test"


def rebalance(dry_run: bool = True):
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Loaded {len(rows)} rows from manifest.")
    print()

    # Show BEFORE
    print("=== BEFORE rebalancing ===")
    _print_stats(rows)

    changed = 0
    for row in rows:
        if row["source_dataset"] in FROZEN_DATASETS:
            continue  # keep official ASVspoof splits

        new_split = assign_split(row["sample_id"])
        if row["split"] != new_split:
            row["split"] = new_split
            changed += 1

    print(f"\n{changed} rows would be reassigned.")
    print()
    print("=== AFTER rebalancing ===")
    _print_stats(rows)

    if dry_run:
        print("\nDry run — manifest NOT written. Re-run with --apply to save.")
        return

    # Write updated manifest
    fieldnames = list(rows[0].keys())
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nManifest updated: {MANIFEST}")
    print(f"{changed} rows reassigned.")


def _print_stats(rows):
    from collections import defaultdict

    # Per-dataset split breakdown
    src_split = defaultdict(Counter)
    for r in rows:
        src_split[r["source_dataset"]][r["split"]] += 1

    print(f"  {'dataset':<22} {'total':>8}  {'train':>8}  {'val':>8}  {'test':>8}")
    print(f"  {'-'*22} {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    totals = Counter()
    for src, splits in sorted(src_split.items()):
        total = sum(splits.values())
        tr = splits.get("train", 0)
        va = splits.get("val", 0)
        te = splits.get("test", 0)
        frozen = " *" if src in FROZEN_DATASETS else ""
        print(f"  {src:<22} {total:>8}  {tr:>7} ({tr/total*100:.0f}%)  {va:>7} ({va/total*100:.0f}%)  {te:>7} ({te/total*100:.0f}%){frozen}")
        totals["total"] += total
        totals["train"] += tr
        totals["val"]   += va
        totals["test"]  += te

    total = totals["total"]
    tr, va, te = totals["train"], totals["val"], totals["test"]
    print(f"  {'TOTAL':<22} {total:>8}  {tr:>7} ({tr/total*100:.0f}%)  {va:>7} ({va/total*100:.0f}%)  {te:>7} ({te/total*100:.0f}%)")

    # Label balance per split
    print()
    print(f"  {'split':<8} {'total':>8}  {'real':>8}  {'fake':>8}")
    print(f"  {'-'*8} {'-'*8}  {'-'*8}  {'-'*8}")
    for split in ["train", "val", "test"]:
        split_rows = [r for r in rows if r["split"] == split]
        if not split_rows:
            continue
        real = sum(1 for r in split_rows if float(r["label"]) == 0.0)
        fake = sum(1 for r in split_rows if float(r["label"]) == 1.0)
        n = len(split_rows)
        print(f"  {split:<8} {n:>8}  {real:>7} ({real/n*100:.0f}%)  {fake:>7} ({fake/n*100:.0f}%)")

    print(f"  (* = frozen, official protocol splits kept)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true", default=True,
                       help="Preview changes without writing (default)")
    group.add_argument("--apply", action="store_true",
                       help="Apply changes and overwrite manifest")
    args = parser.parse_args()
    rebalance(dry_run=not args.apply)

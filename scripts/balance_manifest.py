"""
Balance master_manifest.csv so AI (fake) ≈ human (real) at the source-file level.

Problem: ASVspoof 2019 LA is fake-heavy (108,978 spoof / 12,483 bonafide), so the
full corpus ends up ~4:1 fake:real. Training on this biases the detector.

Strategy:
  - Keep ALL real (bonafide + human-recorded) files.
  - Subsample asvspoof2019 FAKE stratified by attack id (A01-A19) so that every
    spoofing method is still represented uniformly.
  - Keep all non-asvspoof2019 fake sources (asvspoof2021 fake, audiogen) in full —
    they're small.
  - Selection is deterministic via MD5(sample_id) so re-runs are reproducible and
    the subsample is stable across machines.
  - 70/15/15 splits are re-assigned AFTER balancing via the same hash so the
    final splits are stratified by source AND label.

Usage:
    python scripts/balance_manifest.py            # dry-run
    python scripts/balance_manifest.py --apply    # overwrite master_manifest.csv
"""

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "metadata" / "master_manifest.csv"

# Source to subsample + target total after subsampling.
SUBSAMPLE_SOURCE = "asvspoof2019"
SUBSAMPLE_TARGET = 24_000          # total fake asvspoof2019 kept (~1,263 per attack)

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15


def _hash_bucket(key: str, mod: int) -> int:
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % mod


def assign_split(sample_id: str) -> str:
    h = _hash_bucket(sample_id, 100)
    if h < int(TRAIN_RATIO * 100):
        return "train"
    if h < int((TRAIN_RATIO + VAL_RATIO) * 100):
        return "val"
    return "test"


def keep_fake(row: dict, per_attack_target: dict[str, int]) -> bool:
    """Decide whether to keep this asvspoof2019 FAKE row.

    Uses a deterministic hash on sample_id, taking the `target`-th smallest
    hash-valued sample_ids within each attack. Equivalent to: assign a uniform
    random score in [0, group_size) per sample and keep the lowest `target`.
    """
    gen = row["generator"]
    target = per_attack_target.get(gen, 0)
    if target == 0:
        return False
    # Bucket mod large space, keep if bucket < target * (large / group_size).
    # Simpler exact approach: hash -> rank position is not available without
    # enumerating the group, so we use a rate: target / group_size.
    # `per_attack_target` is already {gen: (target, group_size)}.
    tgt, group_size = target
    if group_size == 0:
        return False
    bucket = _hash_bucket(row["sample_id"], group_size)
    return bucket < tgt


def balance(dry_run: bool = True):
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Loaded {len(rows)} rows.")
    print("\n=== BEFORE ===")
    _print_label_stats(rows)

    # 1. Partition rows.
    real_rows = [r for r in rows if float(r["label"]) == 0.0]
    fake_rows = [r for r in rows if float(r["label"]) == 1.0]

    asvspoof19_fake = [r for r in fake_rows if r["source_dataset"] == SUBSAMPLE_SOURCE]
    other_fake      = [r for r in fake_rows if r["source_dataset"] != SUBSAMPLE_SOURCE]

    # 2. Compute per-attack subsample targets.
    attacks = Counter(r["generator"] for r in asvspoof19_fake)
    n_attacks = len(attacks)
    per_attack_target_count = SUBSAMPLE_TARGET // n_attacks   # uniform per attack

    per_attack_target: dict[str, tuple[int, int]] = {
        gen: (min(per_attack_target_count, size), size) for gen, size in attacks.items()
    }

    print(f"\nSubsampling {SUBSAMPLE_SOURCE} fake: {len(asvspoof19_fake)} -> target {SUBSAMPLE_TARGET}")
    print(f"  {n_attacks} attacks, ~{per_attack_target_count} each")

    # 3. Filter fake rows.
    kept_asvspoof19 = [r for r in asvspoof19_fake if keep_fake(r, per_attack_target)]
    new_fake = kept_asvspoof19 + other_fake
    new_rows = real_rows + new_fake

    # Verify per-attack distribution
    kept_by_attack = Counter(r["generator"] for r in kept_asvspoof19)
    print("  per-attack kept:")
    for gen in sorted(kept_by_attack):
        orig = attacks[gen]
        kept = kept_by_attack[gen]
        print(f"    {gen}  {kept:>5} / {orig:>6}  ({kept/orig*100:.1f}%)")

    # 4. Re-assign splits stratified by new sample_ids.
    for row in new_rows:
        row["split"] = assign_split(row["sample_id"])

    print("\n=== AFTER ===")
    _print_label_stats(new_rows)
    _print_split_stats(new_rows)

    if dry_run:
        print("\nDry run — manifest NOT written. Re-run with --apply to save.")
        return

    fieldnames = list(rows[0].keys())
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_rows)

    print(f"\nManifest updated: {MANIFEST}")
    print(f"Source file count: {len(rows)} -> {len(new_rows)}")


def _print_label_stats(rows):
    by_src_label: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        lab = "real" if float(r["label"]) == 0.0 else "fake"
        by_src_label[r["source_dataset"]][lab] += 1

    print(f"  {'source':<20} {'real':>8} {'fake':>8} {'total':>8}")
    print("  " + "-" * 48)
    tr = tf = 0
    for src in sorted(by_src_label):
        r = by_src_label[src].get("real", 0)
        f = by_src_label[src].get("fake", 0)
        tr += r
        tf += f
        print(f"  {src:<20} {r:>8} {f:>8} {r+f:>8}")
    print("  " + "-" * 48)
    print(f"  {'TOTAL':<20} {tr:>8} {tf:>8} {tr+tf:>8}")
    if tr > 0:
        print(f"  fake:real = {tf/tr:.2f}:1")


def _print_split_stats(rows):
    sp_lab: Counter = Counter()
    for r in rows:
        lab = "real" if float(r["label"]) == 0.0 else "fake"
        sp_lab[(r["split"], lab)] += 1
    print(f"\n  {'split':<8} {'real':>8} {'fake':>8} {'total':>8}")
    print("  " + "-" * 36)
    for sp in ["train", "val", "test"]:
        r = sp_lab[(sp, "real")]
        f = sp_lab[(sp, "fake")]
        print(f"  {sp:<8} {r:>8} {f:>8} {r+f:>8}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true", default=True,
                       help="Preview changes without writing (default)")
    group.add_argument("--apply", action="store_true",
                       help="Apply changes and overwrite manifest")
    args = parser.parse_args()
    balance(dry_run=not args.apply)

"""
Pre-training audit of master_manifest_segmented.csv.

Runs seven independent checks and prints a verdict for each. Designed to be
read by a non-expert collaborator before kicking off a long training run.

Checks:
  1. Segment-level leakage       — any full sample_id in >1 split?
  2. Source-file-level leakage   — any <sample_id minus _segNNNN> in >1 split?
  3. Suno/Udio same-prompt leakage — do fake_NNNNN_suno_{0,1} share a split?
  4. Label consistency per source — does each source have a coherent label?
  5. Split ratio per source       — is every source split ~70/15/15?
  6. Class balance per split      — is real/fake ratio comparable in train vs test?
  7. Per-domain class balance    — does every (domain, label) cell have >1 sample per split?

Usage (no arguments — reads from the repo's default manifest location):
    python scripts/audit_manifest.py

Exits with non-zero status if any blocking issue is found; prints a green
summary otherwise. Run this before every new training run.
"""

import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


MANIFEST = Path("data/metadata/master_manifest_segmented.csv")


def load() -> list[dict]:
    if not MANIFEST.exists():
        print(f"FATAL: manifest not found at {MANIFEST}")
        sys.exit(2)
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows):,} rows from {MANIFEST}")
    return rows


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def check_1_segment_level_leakage(rows: list[dict]) -> bool:
    """Any sample_id appearing in more than one split."""
    section("1. Segment-level leakage (sample_id in >1 split)")
    by_sid = defaultdict(set)
    for r in rows:
        by_sid[r["sample_id"]].add(r["split"])
    bad = {k: v for k, v in by_sid.items() if len(v) > 1}
    if not bad:
        print("  PASS — no duplicate sample_id across splits")
        return True
    print(f"  FAIL — {len(bad)} sample_ids span multiple splits")
    for sid, splits in list(bad.items())[:3]:
        print(f"    {sid}  splits={sorted(splits)}")
    return False


def check_2_source_file_leakage(rows: list[dict]) -> bool:
    """Segments from the same source file landing in different splits."""
    section("2. Source-file-level leakage (segments of one clip across splits)")
    by_orig = defaultdict(set)
    by_orig_src = {}
    for r in rows:
        orig = re.sub(r"_seg\d+$", "", r["sample_id"])
        by_orig[orig].add(r["split"])
        by_orig_src[orig] = r["source_dataset"]
    bad = {k: v for k, v in by_orig.items() if len(v) > 1}
    if not bad:
        print("  PASS — every source file's segments stay in one split")
        return True
    by_src = Counter(by_orig_src[k] for k in bad)
    print(f"  FAIL — {len(bad)} source files leak across splits")
    for src, n in by_src.most_common():
        print(f"    {src:<15} {n} leaked source files")
    return False


def check_3_suno_udio_prompt_leakage(rows: list[dict]) -> bool:
    """Suno/Udio filenames are fake_NNNNN_<gen>_{0,1} — two renderings of the
    same prompt. Check whether _0 and _1 land in different splits."""
    section("3. Suno/Udio same-prompt leakage (fake_NNNNN across renderings)")
    # Group by prompt-id: strip trailing _<gen>_<rendering_idx>_segNNNN
    prompt_pattern = re.compile(r"^(?:suno|udio)_fake_(\d+)_(suno|udio)_\d+_seg\d+$")
    by_prompt = defaultdict(set)
    for r in rows:
        if r["source_dataset"] not in ("suno", "udio"):
            continue
        m = prompt_pattern.match(r["sample_id"])
        if not m:
            continue
        key = f'{m.group(2)}_{m.group(1)}'  # e.g. "suno_00011"
        by_prompt[key].add(r["split"])
    bad = {k: v for k, v in by_prompt.items() if len(v) > 1}
    total_prompts = len(by_prompt)
    if not bad:
        print(f"  PASS — all {total_prompts} Suno/Udio prompts confined to one split each")
        return True
    print(f"  WARN — {len(bad)} / {total_prompts} Suno/Udio prompts have "
          f"multiple renderings in different splits")
    print("  (fake_NNNNN_suno_0 and fake_NNNNN_suno_1 are different AI songs from"
          " the same prompt — minor risk of stylistic leakage, not catastrophic)")
    for k, splits in list(bad.items())[:3]:
        print(f"    {k}  splits={sorted(splits)}")
    # This is a soft warning, not a blocker.
    return True


def check_4_label_consistency(rows: list[dict]) -> bool:
    """Each source_dataset should have a consistent label story:
    - Unambiguous 'all real' or 'all fake' sources should have one label.
    - ASVspoof sources legitimately have both."""
    section("4. Label consistency per source")
    by_src_label = defaultdict(Counter)
    for r in rows:
        by_src_label[r["source_dataset"]][r["label"]] += 1
    SHOULD_BE_REAL = {"ljspeech", "musiccaps", "fma_small", "esc50", "fsd50k"}
    SHOULD_BE_FAKE = {"musicgen", "suno", "udio", "audiogen", "audioldm2"}
    MIXED = {"asvspoof2019", "asvspoof2021"}
    all_ok = True
    for src in sorted(by_src_label):
        labels = dict(by_src_label[src])
        if src in SHOULD_BE_REAL:
            if "1.0" in labels and labels["1.0"] > 0:
                print(f"  FAIL — {src} should be 100% real but has {labels['1.0']} fake rows")
                all_ok = False
            else:
                print(f"  ok   {src:<15} {labels}")
        elif src in SHOULD_BE_FAKE:
            if "0.0" in labels and labels["0.0"] > 0:
                print(f"  FAIL — {src} should be 100% fake but has {labels['0.0']} real rows")
                all_ok = False
            else:
                print(f"  ok   {src:<15} {labels}")
        elif src in MIXED:
            print(f"  ok   {src:<15} {labels}  (expected mixed)")
        else:
            print(f"  ?    {src:<15} {labels}  (unknown source; skipping)")
    return all_ok


def check_5_split_ratio_per_source(rows: list[dict]) -> bool:
    """Each source should land near 70/15/15."""
    section("5. Split ratio per source (target 70/15/15)")
    by_src_split = defaultdict(Counter)
    for r in rows:
        by_src_split[r["source_dataset"]][r["split"]] += 1
    all_ok = True
    print(f"  {'source':<15} {'n':>7}  {'train':>5} / {'val':>5} / {'test':>5}")
    for src, c in sorted(by_src_split.items(), key=lambda x: -sum(x[1].values())):
        total = sum(c.values())
        t, v, e = c["train"] / total, c["val"] / total, c["test"] / total
        flag = " "
        # Warn only for gross drift (train < 60% or test < 10%)
        if t < 0.60 or e < 0.10 or v < 0.10:
            flag = "!"
            all_ok = False
        print(f"  {src:<15} {total:>7}  {t:>4.0%} / {v:>4.0%} / {e:>4.0%} {flag}")
    if all_ok:
        print("  PASS — all sources close to 70/15/15")
    else:
        print("  WARN — at least one source significantly off ratio")
    return all_ok


def check_6_class_balance_per_split(rows: list[dict]) -> bool:
    """Real/fake ratio should not drift wildly between train and test."""
    section("6. Real/fake balance per split")
    by_split_label = defaultdict(Counter)
    for r in rows:
        lab = "real" if float(r["label"]) == 0.0 else "fake"
        by_split_label[r["split"]][lab] += 1
    print(f"  {'split':<8} {'real':>8} {'fake':>8} {'real %':>10}")
    ratios = {}
    for split in ("train", "val", "test"):
        c = by_split_label[split]
        real = c["real"]
        fake = c["fake"]
        n = real + fake
        if n == 0:
            print(f"  {split:<8} empty")
            continue
        ratio = real / n
        ratios[split] = ratio
        print(f"  {split:<8} {real:>8} {fake:>8} {ratio:>9.1%}")
    drift = max(ratios.values()) - min(ratios.values()) if ratios else 0
    if drift < 0.03:
        print(f"  PASS — real-class ratio drift between splits: {drift:.1%}")
        return True
    else:
        print(f"  WARN — real-class ratio varies by {drift:.1%} between splits "
              f"(acceptable if <5%, investigate if >10%)")
        return drift < 0.10


def check_7_per_domain_class_balance(rows: list[dict]) -> bool:
    """Every (domain, label) combination should have samples in all three splits."""
    section("7. Per-domain class balance (every bucket must have samples in each split)")
    bucket = defaultdict(Counter)
    for r in rows:
        lab = "real" if float(r["label"]) == 0.0 else "fake"
        bucket[(r["domain"], lab)][r["split"]] += 1
    all_ok = True
    print(f"  {'domain':<12} {'label':<6} {'train':>8} {'val':>8} {'test':>8}  {'min':>6}")
    for (d, lab), c in sorted(bucket.items()):
        t, v, e = c["train"], c["val"], c["test"]
        min_bucket = min(t, v, e)
        flag = " "
        if min_bucket < 50:
            flag = "!"
            all_ok = False
        print(f"  {d:<12} {lab:<6} {t:>8} {v:>8} {e:>8}  {min_bucket:>6} {flag}")
    if all_ok:
        print("  PASS — every bucket has >=50 samples in every split")
    else:
        print("  WARN — at least one (domain,label,split) cell has <50 samples "
              "(too few for reliable per-domain EER)")
    return all_ok


def main() -> int:
    rows = load()
    results = [
        check_1_segment_level_leakage(rows),
        check_2_source_file_leakage(rows),
        check_3_suno_udio_prompt_leakage(rows),
        check_4_label_consistency(rows),
        check_5_split_ratio_per_source(rows),
        check_6_class_balance_per_split(rows),
        check_7_per_domain_class_balance(rows),
    ]
    section("SUMMARY")
    passed = sum(results)
    total = len(results)
    print(f"  {passed}/{total} checks passed")
    if passed == total:
        print("  All clear — safe to train.")
        return 0
    else:
        print("  Fix the FAIL/WARN items above before training (or accept the risk).")
        return 1


if __name__ == "__main__":
    sys.exit(main())

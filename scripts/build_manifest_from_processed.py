"""
Build master_manifest_segmented.csv by walking the Processed/ folder.

Used when preprocessing was already run (by a teammate) and only the
Processed output is available, without the segmented manifest.

Layout expected under --processed-root:
    <root>/<domain>/<real_or_fake>/<source>/<file>.wav
    <root>/Voice/Anh-ASVspoof 2021 voice clips/AI voice/<hash>.wav     (fake)
    <root>/Voice/Anh-ASVspoof 2021 voice clips/Human voice/<hash>.wav  (real)

Splits are assigned deterministically by hashing the source-file stem
(segments from the same source stay together — no cross-segment leakage).
"""

import argparse
import csv
import hashlib
import re
from collections import Counter
from pathlib import Path

SEG_RE = re.compile(r"^(.+)_seg(\d+)$")

DOMAIN_NORMALIZE = {
    "voice": "voice",
    "non_human": "non_human",
    "music": "music",
}

ASVSPOOF_DIR = "Anh-ASVspoof 2021 voice clips"
ASVSPOOF_FAKE_SUBDIR = "AI voice"
ASVSPOOF_REAL_SUBDIR = "Human voice"


def split_for(stem: str, train_frac: float = 0.8, val_frac: float = 0.1) -> str:
    h = int(hashlib.md5(stem.encode("utf-8")).hexdigest(), 16) % 1000
    if h < train_frac * 1000:
        return "train"
    if h < (train_frac + val_frac) * 1000:
        return "val"
    return "test"


def parse_standard(wav: Path, domain_dir: str, bucket: str, source: str) -> dict:
    name = wav.stem
    m = SEG_RE.match(name)
    if m:
        stem, seg_idx = m.group(1), int(m.group(2))
    else:
        stem, seg_idx = name, 0
    label = 1.0 if bucket == "fake" else 0.0
    return {
        "sample_id": f"{source}_{stem}_seg{seg_idx:04d}",
        "file_path": str(wav.resolve()).replace("\\", "/"),
        "label": label,
        "domain": DOMAIN_NORMALIZE.get(domain_dir.lower(), domain_dir.lower()),
        "source_dataset": source,
        "generator": "unknown",
        "split": split_for(stem),
        "seg_index": seg_idx,
        "source_file": stem,
    }


def parse_asvspoof(wav: Path, bucket: str) -> dict:
    # ASVspoof segments are just <hash>.wav; each is its own source file.
    stem = wav.stem
    label = 1.0 if bucket == "fake" else 0.0
    return {
        "sample_id": f"asvspoof2021_{stem}_seg0000",
        "file_path": str(wav.resolve()).replace("\\", "/"),
        "label": label,
        "domain": "voice",
        "source_dataset": "asvspoof2021",
        "generator": "unknown",
        "split": split_for(stem),
        "seg_index": 0,
        "source_file": stem,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", required=True)
    parser.add_argument("--out", default="data/metadata/master_manifest_segmented.csv")
    args = parser.parse_args()

    root = Path(args.processed_root).resolve()
    if not root.exists():
        raise SystemExit(f"Processed root not found: {root}")

    rows = []
    skipped = 0

    for domain_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for bucket_dir in sorted(p for p in domain_dir.iterdir() if p.is_dir()):
            # ASVspoof special case — sits where a real/fake bucket would be.
            if bucket_dir.name == ASVSPOOF_DIR:
                fake_dir = bucket_dir / ASVSPOOF_FAKE_SUBDIR
                real_dir = bucket_dir / ASVSPOOF_REAL_SUBDIR
                for wav in fake_dir.glob("*.wav"):
                    rows.append(parse_asvspoof(wav, "fake"))
                for wav in real_dir.glob("*.wav"):
                    rows.append(parse_asvspoof(wav, "real"))
                continue

            if bucket_dir.name.lower() not in ("real", "fake"):
                skipped += 1
                continue

            bucket = bucket_dir.name.lower()
            for source_dir in sorted(p for p in bucket_dir.iterdir() if p.is_dir()):
                source = source_dir.name
                for wav in source_dir.glob("*.wav"):
                    rows.append(parse_standard(wav, domain_dir.name, bucket, source))

    if not rows:
        raise SystemExit("No .wav files found under Processed/.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    by_split = Counter(r["split"] for r in rows)
    by_source = Counter(r["source_dataset"] for r in rows)
    by_label = Counter(int(r["label"]) for r in rows)
    print(f"Wrote {len(rows)} rows to {out_path}")
    print(f"\n  Split:  {dict(by_split)}")
    print(f"  Source: {dict(by_source)}")
    print(f"  Label:  real={by_label.get(0, 0)}  fake={by_label.get(1, 0)}")
    if skipped:
        print(f"  (skipped {skipped} non-real/fake dirs at domain level)")


if __name__ == "__main__":
    main()

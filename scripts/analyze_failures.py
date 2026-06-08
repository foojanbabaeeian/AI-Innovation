"""
Export worst errors for §7.3 failure-mode analysis.

Reads predictions.npz from evaluate.py and writes a CSV of top false
positives / false negatives per domain (by |score - label| or binary error).

Usage:
    python scripts/analyze_failures.py \\
        --predictions outputs/eval/full_model/predictions.npz \\
        --manifest data/metadata/master_manifest_segmented.csv \\
        --output outputs/eval/full_model/failure_cases.csv \\
        --top-k 10
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_manifest_index(manifest_path: str) -> dict[str, dict]:
    """Map sample_id -> manifest row (file_path, domain, source_dataset)."""
    index: dict[str, dict] = {}
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            index[row["sample_id"]] = row
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="predictions.npz from evaluate.py")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    data = np.load(args.predictions, allow_pickle=False)
    scores = data["scores"].astype(float)
    ratios = data["ratios"].astype(float)
    labels = data["labels"].astype(float)
    domains = data["domains"]
    sources = data["sources"]

    manifest = load_manifest_index(args.manifest)
    n = len(scores)
    binary_gt = (labels >= 0.5).astype(int)
    binary_pred = (scores >= args.threshold).astype(int)
    is_error = binary_gt != binary_pred
    abs_err = np.abs(scores - labels)

    rows_out = []
    for domain in sorted(set(domains.tolist())):
        mask = domains == domain
        idx = np.where(mask)[0]

        # False positives: predicted AI, actually real
        fp_idx = idx[(binary_pred[idx] == 1) & (binary_gt[idx] == 0)]
        fp_rank = fp_idx[np.argsort(-scores[fp_idx])[: args.top_k]]

        # False negatives: predicted real, actually AI
        fn_idx = idx[(binary_pred[idx] == 0) & (binary_gt[idx] == 1)]
        fn_rank = fn_idx[np.argsort(scores[fn_idx])[: args.top_k]]

        # High calibration error regardless of threshold
        cal_idx = idx[np.argsort(-abs_err[idx])[: args.top_k]]

        for kind, ranked in [
            ("false_positive", fp_rank),
            ("false_negative", fn_rank),
            ("high_calibration_error", cal_idx),
        ]:
            for i in ranked:
                sid = None
                # predictions.npz has no sample_id; match by parallel order if manifest
                # was built in same order — use index into filtered test set instead.
                meta = {
                    "domain": str(domains[i]),
                    "source": str(sources[i]),
                    "score": float(scores[i]),
                    "label": float(labels[i]),
                    "abs_error": float(abs_err[i]),
                    "error_type": kind,
                    "test_index": int(i),
                }
                rows_out.append(meta)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["error_type", "domain", "source", "score", "label", "abs_error", "test_index"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    n_err = int(is_error.sum())
    print(f"Test segments: {n}")
    print(f"Binary errors @ {args.threshold}: {n_err} ({100 * n_err / n:.2f}%)")
    print(f"Wrote {len(rows_out)} ranked cases -> {out_path}")


if __name__ == "__main__":
    main()

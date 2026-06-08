"""
Evaluate a trained AI-audio detector checkpoint on the test split.

Produces:
  1. Plain-text summary to stdout.
  2. `metrics.json` with all numbers (per-domain, per-source, overall).
  3. `main_row.tex`   — a LaTeX row for the paper's Table 2
       (Voice EER / Music EER / Non-hum EER / Overall EER).
  4. `per_generator_table.tex` — LaTeX rows for Table 3
       (per-source EER, separated by domain).

Usage:
    # Full model
    python scripts/evaluate.py \\
        --config configs/gpu_local.yaml \\
        --checkpoint outputs/ai_audio_detection/checkpoint_best.pt \\
        --output-dir outputs/eval/full_model \\
        --label "Ours (full)"

    # Ablation run
    python scripts/evaluate.py \\
        --config configs/ablation_spectral_only.yaml \\
        --checkpoint outputs/ablation_spectral/checkpoint_best.pt \\
        --output-dir outputs/eval/spectral_only \\
        --label "Spectral only"

Outputs land in `--output-dir` (default: next to the checkpoint).
Pass `--label "..."` to set the Model column value in the LaTeX row.
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import ManifestAudioDataset
from src.models.fusion_model import MultiBranchFusionModel
from src.utils.config import Config
from src.utils.metrics import compute_eer, compute_metrics


# Canonical domain order for the paper's Table 2 columns.
DOMAIN_ORDER = ["voice", "music", "non_human"]


def load_model(config: Config, checkpoint_path: str, device: torch.device) -> MultiBranchFusionModel:
    """Rebuild the model architecture from config and load the checkpoint weights."""
    model = MultiBranchFusionModel(
        sample_rate=config.audio.sample_rate,
        embed_dim=config.model.embed_dim,
        num_attention_heads=config.model.num_attention_heads,
        num_attention_layers=config.model.num_attention_layers,
        num_classes=config.model.num_classes,
        ssl_model_name=config.model.ssl_model_name,
        freeze_ssl_feature_extractor=config.model.freeze_ssl_feature_extractor,
        dropout=config.model.dropout,
        disable_branches=list(config.model.disable_branches),
        fusion_method=config.model.fusion_method,
        ssl_layer_mode=config.model.ssl_layer_mode,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"  epoch={ckpt.get('epoch', '?')}  best_eer={ckpt.get('best_eer', '?')}")
    return model


def run_inference(
    model: MultiBranchFusionModel,
    dataset: ManifestAudioDataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict:
    """Run the model over the full dataset, return predictions and metadata."""
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                        shuffle=False, pin_memory=(device.type == "cuda"))

    all_scores, all_logits, all_ratios, all_labels = [], [], [], []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            waveform = batch["waveform"].to(device, non_blocking=True)
            out = model(waveform)
            all_scores.append(out["regression_score"].detach().cpu())
            all_logits.append(out["class_logits"].detach().cpu())
            all_ratios.append(batch["ai_ratio"])
            all_labels.append(batch["class_label"])
            if (i + 1) % 50 == 0:
                print(f"  batch {i+1}/{len(loader)}")

    scores = torch.cat(all_scores).numpy().reshape(-1)
    logits = torch.cat(all_logits).numpy()
    ratios = torch.cat(all_ratios).numpy().reshape(-1)
    labels = torch.cat(all_labels).numpy().reshape(-1).astype(int)

    # Metadata in the same order as the dataset iteration.
    domains = np.array([r.get("domain", "unknown") for r in dataset.records])
    sources = np.array([r.get("source_dataset", "unknown") for r in dataset.records])

    assert len(scores) == len(domains) == len(sources), \
        f"length mismatch: scores={len(scores)} domains={len(domains)} sources={len(sources)}"

    return {
        "scores": scores,
        "logits": logits,
        "ratios": ratios,
        "labels": labels,
        "domains": domains,
        "sources": sources,
    }


def subset_metrics(pred: dict, mask: np.ndarray) -> dict | None:
    """Compute metrics on a subset defined by `mask`. Returns None if too few samples or one-class."""
    if mask.sum() < 10:
        return None
    ratios_sub = pred["ratios"][mask]
    # EER is undefined if only one class is present in the subset.
    binary = (ratios_sub >= 0.2).astype(int)
    if len(np.unique(binary)) < 2:
        return {"note": "single-class subset — EER undefined", "n": int(mask.sum())}
    return compute_metrics(
        regression_scores=pred["scores"][mask],
        class_logits=pred["logits"][mask],
        ai_ratios=ratios_sub,
        class_labels=pred["labels"][mask],
    )


def aggregate(pred: dict) -> dict:
    """Build the overall / per-domain / per-source metrics dictionary."""
    results = {"n_samples": int(len(pred["scores"]))}

    # Overall
    results["overall"] = compute_metrics(
        pred["scores"], pred["logits"], pred["ratios"], pred["labels"]
    )

    # Per-domain
    per_domain = {}
    for d in sorted(np.unique(pred["domains"])):
        m = subset_metrics(pred, pred["domains"] == d)
        if m is not None:
            per_domain[d] = {"n": int((pred["domains"] == d).sum()), **m}
    results["per_domain"] = per_domain

    # Per-source (domain-qualified so the paper can group them)
    per_source = {}
    for s in sorted(np.unique(pred["sources"])):
        mask = pred["sources"] == s
        domain = pred["domains"][mask][0]  # all same source => same domain
        m = subset_metrics(pred, mask)
        if m is not None:
            per_source[s] = {"domain": str(domain), "n": int(mask.sum()), **m}
    results["per_source"] = per_source

    return results


def format_stdout_summary(results: dict, label: str) -> str:
    """Human-readable summary table."""
    lines = ["=" * 72, f"  EVALUATION RESULTS — {label}", "=" * 72,
             f"  N samples: {results['n_samples']}", ""]

    # Overall
    o = results["overall"]
    lines += ["── Overall ──"]
    for k in ("binary/eer", "binary/auc_roc", "binary/avg_precision",
              "classification/accuracy", "classification/accuracy_real",
              "classification/accuracy_AI", "regression/mae"):
        if k in o:
            lines.append(f"    {k:<38}  {o[k]:.4f}")
    lines.append("")

    # Per-domain
    lines += ["── Per-domain ──",
              f"    {'domain':<12} {'n':>6}  {'EER':>8}  {'AUC':>8}  {'acc':>8}"]
    for d in DOMAIN_ORDER + [x for x in results["per_domain"] if x not in DOMAIN_ORDER]:
        if d not in results["per_domain"]:
            continue
        r = results["per_domain"][d]
        if "note" in r:
            lines.append(f"    {d:<12} {r['n']:>6}  {r['note']}")
            continue
        lines.append(
            f"    {d:<12} {r['n']:>6}  "
            f"{r.get('binary/eer', float('nan')):>8.4f}  "
            f"{r.get('binary/auc_roc', float('nan')):>8.4f}  "
            f"{r.get('classification/accuracy', float('nan')):>8.4f}"
        )
    lines.append("")

    # Per-source
    lines += ["── Per-source ──",
              f"    {'source':<20} {'domain':<10} {'n':>6}  {'EER':>8}  {'AUC':>8}"]
    for s in sorted(results["per_source"], key=lambda k: (results["per_source"][k]["domain"], k)):
        r = results["per_source"][s]
        if "note" in r:
            lines.append(f"    {s:<20} {r['domain']:<10} {r['n']:>6}  {r['note']}")
            continue
        lines.append(
            f"    {s:<20} {r['domain']:<10} {r['n']:>6}  "
            f"{r.get('binary/eer', float('nan')):>8.4f}  "
            f"{r.get('binary/auc_roc', float('nan')):>8.4f}"
        )

    lines.append("=" * 72)
    return "\n".join(lines)


def latex_main_row(results: dict, label: str) -> str:
    """One row for the main-results table: Model & Voice & Music & Non-hum & Overall \\\\"""
    def fmt(domain: str) -> str:
        r = results["per_domain"].get(domain)
        if r is None or "binary/eer" not in r:
            return r"\NUM"
        return f"{r['binary/eer'] * 100:.2f}"

    overall_eer = results["overall"].get("binary/eer", float("nan")) * 100
    # Escape LaTeX-special chars in the label.
    label_tex = label.replace("&", r"\&").replace("_", r"\_")
    return f"{label_tex} & {fmt('voice')} & {fmt('music')} & {fmt('non_human')} & {overall_eer:.2f} \\\\"


def latex_per_generator_rows(results: dict, label: str) -> str:
    """Per-source table. Groups fake-only sources under domain with \\midrule separators."""
    FAKE_GENERATORS = {  # only fake-producing sources for per-generator detection
        "voice":     ["asvspoof2019", "asvspoof2021"],
        "music":     ["musicgen", "suno", "udio"],
        "non_human": ["audiogen", "audioldm2"],
    }
    lines = []
    for domain, srcs in FAKE_GENERATORS.items():
        first_in_domain = True
        for src in srcs:
            r = results["per_source"].get(src)
            if r is None:
                continue
            n = r.get("n", "-")
            eer = r.get("binary/eer", None)
            eer_str = f"{eer * 100:.2f}" if eer is not None else r"\NUM"
            row_label = domain if first_in_domain else ""
            lines.append(f"{row_label} & {src} & {n} & {eer_str} \\\\")
            first_in_domain = False
        lines.append(r"\midrule")
    # Remove trailing \midrule
    if lines and lines[-1] == r"\midrule":
        lines.pop()
    header = f"% Per-source EER from model: {label}"
    return "\n".join([header] + lines)


def compute_leaked_source_files(manifest_path: str) -> set[str]:
    """Find source files whose segments span multiple splits.

    A "source file" is a manifest entry keyed by ``sample_id`` with the trailing
    ``_segNNNN`` suffix stripped. If segments sharing the same stripped id land
    in more than one split, the test segments are effectively contaminated by
    train/val siblings — any model can recognize them trivially.

    Returns the set of stripped ids that leak across splits.
    """
    by_orig: dict[str, set[str]] = defaultdict(set)
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            orig = re.sub(r"_seg\d+$", "", row["sample_id"])
            by_orig[orig].add(row["split"])
    return {orig for orig, splits in by_orig.items() if len(splits) > 1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True, type=str,
                        help="Config YAML the model was trained with.")
    parser.add_argument("--checkpoint", required=True, type=str,
                        help="Path to .pt file (typically checkpoint_best.pt).")
    parser.add_argument("--output-dir", default=None, type=str,
                        help="Where to write metrics.json + LaTeX fragments. "
                             "Default: sibling of checkpoint.")
    parser.add_argument("--label", default="Ours (full)", type=str,
                        help="Model name for the LaTeX row (e.g. 'Spectral only').")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override config batch_size.")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap test-set size for smoke testing.")
    parser.add_argument("--manifest", default=None,
                        help="Override config manifest path (e.g. segmented manifest).")
    parser.add_argument("--sources", nargs="+", default=None,
                        help="Whitelist of source_dataset values. "
                             "Cross-dataset: train on ASVspoof19, evaluate with "
                             "'--sources asvspoof2021' for generalization EER.")
    parser.add_argument("--exclude-leaked", action="store_true",
                        help="Exclude test segments whose source file also has segments in "
                             "the train or val splits. Use this when a prior manifest build "
                             "incorrectly hashed by segment-stem instead of source-file, "
                             "causing sibling segments to cross split boundaries. Reports "
                             "honest held-out EER without requiring a retrain.")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Resolve output directory
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Outputs: {out_dir}")

    # Build test dataset. CLI --sources takes precedence over config.data.sources.
    manifest = args.manifest or config.data.manifest_path
    source_filter = args.sources or (list(config.data.sources) if config.data.sources else None)
    if source_filter:
        print(f"Source filter: {source_filter}")
    dataset = ManifestAudioDataset(
        manifest_path=manifest,
        data_root=config.data.data_root,
        split="test",
        target_sr=config.audio.sample_rate,
        segment_length=config.data.segment_length,
        max_samples=args.max_samples,
        sources=source_filter,
    )
    print(f"Test set: {len(dataset)} segments")

    # Optional: drop test segments whose source file also appears in train/val.
    # Fixes the "my EER is suspiciously low" failure mode when a prior manifest
    # build incorrectly hashed by segment-stem and sibling segments crossed
    # splits. Reports honest held-out EER without requiring a retrain.
    if args.exclude_leaked:
        leaked = compute_leaked_source_files(manifest)
        before = len(dataset.records)
        dataset.records = [
            r for r in dataset.records
            if re.sub(r"_seg\d+$", "", r["sample_id"]) not in leaked
        ]
        dropped = before - len(dataset.records)
        print(f"--exclude-leaked: dropped {dropped} test segments "
              f"(from {len(leaked)} leaked source files across the full manifest); "
              f"{len(dataset.records)} test segments remain.")

    batch_size = args.batch_size or config.data.batch_size
    model = load_model(config, args.checkpoint, device)

    print("\nRunning inference...")
    pred = run_inference(model, dataset, batch_size, config.data.num_workers, device)

    print("Aggregating metrics...")
    results = aggregate(pred)

    # Write artifacts
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2, default=float))
    (out_dir / "main_row.tex").write_text(latex_main_row(results, args.label) + "\n")
    (out_dir / "per_generator_rows.tex").write_text(latex_per_generator_rows(results, args.label) + "\n")

    # Save raw predictions for ensembling / later analysis
    np.savez(out_dir / "predictions.npz",
             scores=pred["scores"], logits=pred["logits"],
             ratios=pred["ratios"], labels=pred["labels"],
             domains=pred["domains"], sources=pred["sources"])

    # Print summary
    print(format_stdout_summary(results, args.label))
    print(f"\n✓ Wrote: {out_dir}/metrics.json")
    print(f"✓ Wrote: {out_dir}/main_row.tex")
    print(f"✓ Wrote: {out_dir}/per_generator_rows.tex")
    print(f"✓ Wrote: {out_dir}/predictions.npz")


if __name__ == "__main__":
    main()

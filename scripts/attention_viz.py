"""
Attention-weight visualization for the fusion layer — paper Figure §7.1.

Extracts the softmax attention matrices produced inside each
CrossBranchAttention layer of the trained model, averages them across
heads + layers + test samples, and reports per-domain means as a grouped
bar chart plus a CSV table.

Interpretation: for each domain (voice / music / non-human), how much
attention does the fused representation place on each of the three
branches (spectral / SSL / raw)? High values mean the corresponding
branch dominates the decision for that domain — directly supports the
paper's claim that attention fusion adapts per-input.

Usage:
    python scripts/attention_viz.py \\
        --config     configs/gpu_local.yaml \\
        --checkpoint outputs/ai_audio_detection/checkpoint_best.pt \\
        --output-dir outputs/figures/attention

Outputs:
    attention_per_domain.pdf    Paper figure (vector, column-width).
    attention_per_domain.png    Preview (dpi=200).
    attention_per_domain.csv    Raw numbers + std across samples.
    attention_per_source.csv    Finer-grained breakdown per source_dataset.

Only runs on checkpoints with the full 3-branch attention-fusion architecture
(i.e., fusion_method='attention' and no branches disabled).
"""

import argparse
import csv as csv_mod
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.fusion_model import MultiBranchFusionModel
from src.utils.config import Config
from src.data.dataset import ManifestAudioDataset


BRANCH_LABELS = ["spectral", "SSL", "raw"]
BRANCH_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]
DOMAIN_ORDER = ["voice", "music", "non_human"]


def load_model(config: Config, checkpoint_path: str, device: torch.device) -> MultiBranchFusionModel:
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

    if model.fusion_method != "attention":
        raise RuntimeError(
            f"This script requires fusion_method='attention' but got "
            f"{model.fusion_method!r}. Use the full attention-fusion checkpoint."
        )
    if len(model.active_branches) < 2:
        raise RuntimeError(
            "Attention visualization needs at least 2 active branches; "
            f"this checkpoint has {model.active_branches}."
        )
    if len(model.attention_layers) == 0:
        raise RuntimeError("Model has no attention layers (unexpected).")
    return model


def register_attention_hooks(model: MultiBranchFusionModel) -> tuple[list, list]:
    """Hook every CrossBranchAttention's internal nn.MultiheadAttention.

    Returns:
        (captured, handles) — captured is a list that the hooks append into;
        handles should be removed after inference.
    """
    captured: list[torch.Tensor] = []
    handles = []

    def make_hook(layer_idx: int):
        def hook(_module, _inp, output):
            # nn.MultiheadAttention returns (attn_output, attn_weights)
            # attn_weights shape (batch, Q, K) averaged over heads by default
            # since need_weights=True and average_attn_weights=True are defaults.
            weights = output[1]
            captured.append(weights.detach().cpu())
        return hook

    for i, layer in enumerate(model.attention_layers):
        h = layer.attn.register_forward_hook(make_hook(i))
        handles.append(h)
    return captured, handles


def run_once(
    model: MultiBranchFusionModel,
    dataset: ManifestAudioDataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict:
    """Run the full test set through the model, collect per-sample attention.

    Returns dict with:
        attn_matrices: (N, n_branches, n_branches) — mean over attention layers
        domains:        (N,) strings
        sources:        (N,) strings
        labels:         (N,) floats in [0, 1]
    """
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                        shuffle=False, pin_memory=(device.type == "cuda"))

    captured, handles = register_attention_hooks(model)
    per_sample_mats = []  # list of (n_branches, n_branches) averaged per sample

    try:
        with torch.no_grad():
            for i, batch in enumerate(loader):
                # reset per-batch capture
                captured.clear()
                waveform = batch["waveform"].to(device, non_blocking=True)
                _ = model(waveform)

                # captured now has len == len(model.attention_layers),
                # each shape (batch, Q, K).
                stacked = torch.stack(captured, dim=0)   # (L, batch, Q, K)
                mean_over_layers = stacked.mean(dim=0)    # (batch, Q, K)

                for row_idx in range(mean_over_layers.shape[0]):
                    per_sample_mats.append(mean_over_layers[row_idx].numpy())

                if (i + 1) % 50 == 0:
                    print(f"  batch {i+1}/{len(loader)}")
    finally:
        for h in handles:
            h.remove()

    attn_matrices = np.stack(per_sample_mats, axis=0)  # (N, Q, K)
    domains = np.array([r["domain"] for r in dataset.records])
    sources = np.array([r["source_dataset"] for r in dataset.records])
    labels = np.array([float(r["label"]) for r in dataset.records])
    return {
        "attn_matrices": attn_matrices,
        "domains": domains,
        "sources": sources,
        "labels": labels,
    }


def compute_per_domain(result: dict, active_branches: tuple) -> dict:
    """Collapse per-sample matrices to per-domain (mean, std) over branches.

    The 'attention received by each key branch' metric is the column-mean of
    the attention matrix (mean over queries) — i.e., on average, how much of
    the attention budget is spent on each branch.
    """
    mats = result["attn_matrices"]  # (N, n, n) where n = num active branches
    domains = result["domains"]
    n_active = mats.shape[1]

    # Per-sample column-mean → (N, n_active)
    per_sample = mats.mean(axis=1)

    per_domain: dict[str, dict[str, np.ndarray]] = {}
    for d in sorted(np.unique(domains)):
        mask = domains == d
        if mask.sum() == 0:
            continue
        vals = per_sample[mask]  # (n_domain_samples, n_active)
        per_domain[d] = {
            "n": int(mask.sum()),
            "mean": vals.mean(axis=0),
            "std": vals.std(axis=0),
        }
    return per_domain


def compute_per_source(result: dict) -> dict:
    """Same as per-domain but grouped by source_dataset (for CSV only)."""
    mats = result["attn_matrices"]
    sources = result["sources"]
    per_sample = mats.mean(axis=1)
    per_source = {}
    for s in sorted(np.unique(sources)):
        mask = sources == s
        if mask.sum() == 0:
            continue
        vals = per_sample[mask]
        per_source[s] = {
            "n": int(mask.sum()),
            "mean": vals.mean(axis=0),
            "std": vals.std(axis=0),
        }
    return per_source


def plot_grouped_bars(per_domain: dict, active_branches: tuple, out_pdf: Path, out_png: Path) -> None:
    """Paper figure: one group per domain, one bar per branch within each group."""
    domains = [d for d in DOMAIN_ORDER if d in per_domain]
    x = np.arange(len(domains))
    n = len(active_branches)
    width = 0.80 / n

    fig, ax = plt.subplots(figsize=(4.8, 3.0), constrained_layout=True)

    for j, branch in enumerate(active_branches):
        means = np.array([per_domain[d]["mean"][j] for d in domains])
        stds = np.array([per_domain[d]["std"][j] for d in domains])
        offset = (j - (n - 1) / 2) * width
        label = BRANCH_LABELS[["spectral", "ssl", "rawnet"].index(branch)]
        color = BRANCH_COLORS[["spectral", "ssl", "rawnet"].index(branch)]
        ax.bar(x + offset, means, width, yerr=stds, label=label,
               color=color, ecolor="#555555", capsize=2)

    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.set_ylabel("Mean attention weight")
    ax.set_ylim(0.0, None)
    ax.axhline(1.0 / n, color="gray", linestyle=":", linewidth=0.8,
               label=f"uniform (1/{n})")
    ax.legend(fontsize=8, loc="best", frameon=False)

    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def write_csv(per_group: dict, active_branches: tuple, out_path: Path, group_key: str) -> None:
    """CSV with one row per (group, branch) pair including mean and std."""
    branch_names = [
        BRANCH_LABELS[["spectral", "ssl", "rawnet"].index(b)] for b in active_branches
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv_mod.writer(f)
        w.writerow([group_key, "n_samples", "branch", "mean_attention", "std_attention"])
        for group, stats in per_group.items():
            for j, branch in enumerate(branch_names):
                w.writerow([group, stats["n"], branch,
                            f"{stats['mean'][j]:.6f}", f"{stats['std'][j]:.6f}"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--output-dir", type=str, default="outputs/figures/attention")
    parser.add_argument("--manifest",   type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap test-set size for smoke testing.")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = args.manifest or config.data.manifest_path
    dataset = ManifestAudioDataset(
        manifest_path=manifest,
        data_root=config.data.data_root,
        split="test",
        target_sr=config.audio.sample_rate,
        segment_length=config.data.segment_length,
        max_samples=args.max_samples,
    )
    print(f"Test set: {len(dataset)} segments")

    model = load_model(config, args.checkpoint, device)
    print(f"Model: active_branches={model.active_branches} fusion={model.fusion_method}")

    batch_size = args.batch_size or config.data.batch_size
    result = run_once(model, dataset, batch_size, config.data.num_workers, device)

    per_domain = compute_per_domain(result, model.active_branches)
    per_source = compute_per_source(result)

    # Plot
    out_pdf = out_dir / "attention_per_domain.pdf"
    out_png = out_dir / "attention_per_domain.png"
    plot_grouped_bars(per_domain, model.active_branches, out_pdf, out_png)

    # CSVs
    write_csv(per_domain, model.active_branches,
              out_dir / "attention_per_domain.csv", "domain")
    write_csv(per_source, model.active_branches,
              out_dir / "attention_per_source.csv", "source")

    # Short stdout summary
    print("\nPer-domain mean attention (row sums may not be 1 across layers):")
    print(f"  {'domain':<12} {'n':>6}  " + "  ".join(f"{b:>8}" for b in model.active_branches))
    for d in [x for x in DOMAIN_ORDER if x in per_domain]:
        stats = per_domain[d]
        vals = "  ".join(f"{v:>8.4f}" for v in stats["mean"])
        print(f"  {d:<12} {stats['n']:>6}  {vals}")

    print(f"\n✓ Wrote {out_pdf}")
    print(f"✓ Wrote {out_dir / 'attention_per_domain.csv'}")
    print(f"✓ Wrote {out_dir / 'attention_per_source.csv'}")


if __name__ == "__main__":
    main()

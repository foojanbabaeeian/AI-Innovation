"""
Generate paper/report figures for the AI audio detector.

Produces three PNG + PDF pairs under docs/figures/:
    1. dataset_composition.{png,pdf}  — segment counts by domain & label (for Data section)
    2. architecture.{png,pdf}          — three-branch late-fusion diagram (for AI Models section)
    3. pipeline_overview.{png,pdf}     — data pipeline flow (for Data section / overview)
    4. deployment.{png,pdf}            — browser extension + backend flow (for Deployment section)

PNGs are sized for a Google Docs paste at column width. PDFs are vector, for LaTeX/paper.

Usage:
    python scripts/generate_figures.py
    python scripts/generate_figures.py --only dataset
    python scripts/generate_figures.py --output-dir some/where

Requires: matplotlib.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


# Colour palette — colorblind-friendly, consistent across all figures.
COLOR_REAL = "#1f77b4"   # blue
COLOR_FAKE = "#d62728"   # red
COLOR_NEUTRAL = "#7f7f7f"
COLOR_SPEC = "#2ca02c"   # green  (Branch 1: spectral)
COLOR_SSL  = "#ff7f0e"   # orange (Branch 2: SSL)
COLOR_RAW  = "#9467bd"   # purple (Branch 3: raw)
COLOR_FUSE = "#17becf"   # teal   (fusion)
COLOR_BG_LIGHT = "#f0f0f0"


# ── Figure 1: Dataset composition ─────────────────────────────────────────────

def make_dataset_composition(out_png: Path, out_pdf: Path) -> None:
    """Horizontal stacked bar of segment counts per domain × label."""
    domains = ["Voice", "Music", "Non-human"]
    # Counts from the final segmented manifest (~196K total)
    real = np.array([33, 20, 36])  # K segments
    fake = np.array([43, 34, 30])

    fig, ax = plt.subplots(figsize=(5.2, 2.4), constrained_layout=True)

    y = np.arange(len(domains))
    bars_r = ax.barh(y, real, color=COLOR_REAL, label="Real", edgecolor="white")
    bars_f = ax.barh(y, fake, left=real, color=COLOR_FAKE, label="AI-generated", edgecolor="white")

    ax.set_yticks(y)
    ax.set_yticklabels(domains)
    ax.set_xlabel("Segments (thousands)")
    ax.set_xlim(0, 90)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.25, linestyle=":")

    # Annotate totals at the end of each bar
    totals = real + fake
    for idx, total in enumerate(totals):
        ax.text(total + 1.2, idx, f"{total}K",
                va="center", fontsize=9, color="#444")

    # Annotate inside bars
    for rect, val in zip(bars_r, real):
        if val > 4:
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_y() + rect.get_height() / 2,
                    f"{val}K", ha="center", va="center", color="white", fontsize=8)
    for rect, val in zip(bars_f, fake):
        if val > 4:
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_y() + rect.get_height() / 2,
                    f"{val}K", ha="center", va="center", color="white", fontsize=8)

    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.set_title("Dataset composition — 196K segments across 3 domains",
                 fontsize=10, loc="left", pad=6)

    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# ── Figure 2: Model architecture ──────────────────────────────────────────────

def _box(ax, x, y, w, h, text, color, text_color="black", fontsize=9, ha="center"):
    """Draw a rounded rectangle with text inside."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.1, edgecolor="#333333", facecolor=color,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text,
            ha=ha, va="center", fontsize=fontsize, color=text_color, wrap=True)


def _arrow(ax, x1, y1, x2, y2, style="-|>", color="#333"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=12,
        linewidth=1.1, color=color,
    ))


def make_architecture(out_png: Path, out_pdf: Path) -> None:
    """Three-branch late-fusion architecture diagram."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Input waveform
    _box(ax, 4.2, 6.0, 1.6, 0.55,
         "Waveform x ∈ ℝ⁶⁴⁰⁰⁰\n(mono 16 kHz, 4 s)",
         color="#e8e8e8", fontsize=8)

    # Three branches (row at y=4)
    _box(ax, 0.3, 4.1, 2.6, 1.1,
         "Branch 1 — Spectral\nMelSpec(128) → ResNet\n(32→64→128→256)",
         color=COLOR_SPEC, text_color="white", fontsize=8)
    _box(ax, 3.7, 4.1, 2.6, 1.1,
         "Branch 2 — SSL\nfrozen WavLM-base+\nlayer-weighted pooling",
         color=COLOR_SSL, text_color="white", fontsize=8)
    _box(ax, 7.1, 4.1, 2.6, 1.1,
         "Branch 3 — Raw\n70 SincNet filters\n→ 3× ResBlock1D",
         color=COLOR_RAW, text_color="white", fontsize=8)

    # Branch outputs (embeddings)
    _box(ax, 0.7, 3.15, 1.8, 0.45, "z_spec ∈ ℝ¹²⁸",
         color="#f0f0f0", fontsize=8)
    _box(ax, 4.1, 3.15, 1.8, 0.45, "z_ssl ∈ ℝ¹²⁸",
         color="#f0f0f0", fontsize=8)
    _box(ax, 7.5, 3.15, 1.8, 0.45, "z_raw ∈ ℝ¹²⁸",
         color="#f0f0f0", fontsize=8)

    # Arrows waveform → branches
    for cx in (1.6, 5.0, 8.4):
        _arrow(ax, 5.0, 6.0, cx, 5.2)
    # branches → embeddings
    for x in (1.6, 5.0, 8.4):
        _arrow(ax, x, 4.1, x, 3.6)

    # Fusion (attention)
    _box(ax, 2.8, 2.15, 4.4, 0.7,
         "Attention Fusion (2-layer, 4-head)\nreweights branches per sample",
         color=COLOR_FUSE, text_color="white", fontsize=9)
    # embeddings → fusion
    for x in (1.6, 5.0, 8.4):
        _arrow(ax, x, 3.15, 5.0, 2.85)

    # Heads
    _box(ax, 1.0, 0.9, 3.6, 0.75,
         "Regression head → σ(·)\nscore ∈ [0,1] = P(AI)",
         color="#e8f4ff", fontsize=9)
    _box(ax, 5.4, 0.9, 3.6, 0.75,
         "Classification head\nlogits ∈ ℝ³ {real, mixed, AI}",
         color="#fff0e8", fontsize=9)
    # fusion → heads
    _arrow(ax, 4.2, 2.15, 2.8, 1.65)
    _arrow(ax, 5.8, 2.15, 7.2, 1.65)

    # Loss note at bottom
    ax.text(5.0, 0.25,
            "DualHeadLoss = MSE(score, ai_ratio) + 0.5 · CE(logits, class_label)",
            ha="center", fontsize=8.5, style="italic", color="#444")

    # Title
    ax.text(5.0, 6.85, "Three-Branch Late-Fusion Architecture",
            ha="center", fontsize=11, fontweight="bold")

    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# ── Figure 3: Data pipeline overview ──────────────────────────────────────────

def make_pipeline(out_png: Path, out_pdf: Path) -> None:
    """Left-to-right data flow: raw → register → preprocess → segmented → train/eval."""
    fig, ax = plt.subplots(figsize=(6.8, 2.0), constrained_layout=True)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.2)
    ax.axis("off")

    stages = [
        (0.1, "raw audio\n(WAV/MP3)", "#e8e8e8"),
        (2.0, "register_\ngenerated.py", "#dde9f5"),
        (3.9, "master_manifest\n.csv", "#f0f0f0"),
        (5.8, "preprocess_\nsegments.py", "#dde9f5"),
        (7.7, "master_manifest_\nsegmented.csv\n(~196K rows)", "#f0f0f0"),
    ]
    w, h = 1.6, 1.05
    for x, txt, color in stages:
        _box(ax, x, 0.7, w, h, txt, color=color, fontsize=8)

    # Arrows
    for i in range(len(stages) - 1):
        x1 = stages[i][0] + w
        x2 = stages[i + 1][0]
        _arrow(ax, x1, 0.7 + h / 2, x2, 0.7 + h / 2)

    # Labels under arrows
    labels = ["walk +\nlog", "row per\nsource file", "mono, 16/44.1 kHz\nEBU R128, 4-s windows",
              "row per\nsegment"]
    for i, lab in enumerate(labels):
        x1 = stages[i][0] + w
        x2 = stages[i + 1][0]
        ax.text((x1 + x2) / 2, 0.45, lab, ha="center", fontsize=7, color="#666",
                style="italic")

    # Title
    ax.text(5.0, 1.95, "Data Pipeline — two-stage manifest",
            ha="center", fontsize=10.5, fontweight="bold")

    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# ── Figure 4: Deployment architecture ─────────────────────────────────────────

def make_deployment(out_png: Path, out_pdf: Path) -> None:
    """Browser extension ↔ FastAPI backend ↔ model."""
    fig, ax = plt.subplots(figsize=(6.8, 3.2), constrained_layout=True)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    # Client-side: browser / extension
    client_box = FancyBboxPatch(
        (0.2, 0.6), 3.8, 2.9,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.0, edgecolor="#333", facecolor="#fafafa",
    )
    ax.add_patch(client_box)
    ax.text(2.1, 3.30, "Chrome Extension (MV3)", ha="center",
            fontsize=9.5, fontweight="bold", color="#333")
    _box(ax, 0.4, 2.55, 3.4, 0.55, "Popup (status + threshold)",
         color="#e8f4ff", fontsize=8)
    _box(ax, 0.4, 1.85, 3.4, 0.55, "Service Worker\n(tabCapture → offscreen)",
         color="#e8f4ff", fontsize=8)
    _box(ax, 0.4, 0.9, 3.4, 0.85,
         "Offscreen Doc:\nAudioContext → AudioWorklet\n(mono float32 frames)",
         color="#e8f4ff", fontsize=8)

    # Server-side: FastAPI backend
    server_box = FancyBboxPatch(
        (6.0, 0.6), 3.8, 2.9,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.0, edgecolor="#333", facecolor="#fafafa",
    )
    ax.add_patch(server_box)
    ax.text(7.9, 3.30, "FastAPI Backend", ha="center",
            fontsize=9.5, fontweight="bold", color="#333")
    _box(ax, 6.2, 2.55, 3.4, 0.55, "WebSocket /ws  (Pydantic msgs)",
         color="#fff0e8", fontsize=8)
    _box(ax, 6.2, 1.85, 3.4, 0.55, "DetectionSession (ring buffer ~4 s)",
         color="#fff0e8", fontsize=8)
    _box(ax, 6.2, 0.9, 3.4, 0.85,
         "AsyncDetector\n(PyTorch model on BG thread)\nscore smoothing + hysteresis",
         color="#fff0e8", fontsize=8)

    # Bidirectional arrow
    _arrow(ax, 4.0, 2.05, 6.0, 2.05)
    _arrow(ax, 6.0, 1.75, 4.0, 1.75)
    ax.text(5.0, 2.25, "audio frames\n(float32 PCM)",
            ha="center", fontsize=7.5, color="#666", style="italic")
    ax.text(5.0, 1.45, "detections /\nalerts",
            ha="center", fontsize=7.5, color="#666", style="italic")

    # Title
    ax.text(5.0, 3.80, "Deployment — tab audio → inference → popup",
            ha="center", fontsize=10.5, fontweight="bold")

    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="docs/figures",
                        help="Where to write the PNGs and PDFs.")
    parser.add_argument("--only", choices=["dataset", "architecture", "pipeline", "deployment"],
                        default=None, help="Generate only one figure.")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    figs = {
        "dataset":      ("dataset_composition", make_dataset_composition),
        "architecture": ("architecture",        make_architecture),
        "pipeline":     ("pipeline_overview",   make_pipeline),
        "deployment":   ("deployment",          make_deployment),
    }

    to_run = {args.only: figs[args.only]} if args.only else figs
    for key, (name, fn) in to_run.items():
        png = out / f"{name}.png"
        pdf = out / f"{name}.pdf"
        print(f"→ {name}.png + .pdf")
        fn(png, pdf)

    print(f"\n✓ Wrote {len(to_run)} figure(s) to {out}/")


if __name__ == "__main__":
    main()

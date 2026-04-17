"""
GradCAM visualization for the spectral branch — produces paper Figure §7.2.

For each input clip, computes a time-frequency heatmap highlighting regions
that drive the AI-detection prediction. Saves one figure per clip as PDF
(vector, paper-ready) and PNG (quick preview).

Method: Grad-CAM of the regression head output w.r.t. the last conv block of
SpectralBranch (``spectral_branch.layer3``). Feature maps are weighted by
channel-wise global-average gradient, summed, ReLU'd, normalized to [0,1],
and bilinearly upsampled to the mel-spectrogram resolution. Standard Grad-CAM
[Selvaraju et al., 2017] applied to P(AI).

Usage:
    # Auto-pick 3 representative clips (1 per domain) from the test split
    python scripts/gradcam.py \\
        --config     configs/gpu_local.yaml \\
        --checkpoint outputs/ai_audio_detection/checkpoint_best.pt \\
        --output-dir outputs/figures/gradcam \\
        --auto-pick

    # Explicit sample IDs (from master_manifest_segmented.csv)
    python scripts/gradcam.py \\
        --config     configs/gpu_local.yaml \\
        --checkpoint outputs/ai_audio_detection/checkpoint_best.pt \\
        --sample-ids musicgen_musicgen_00017_seg0001 \\
                     ljspeech_LJ001-0042_seg0000 \\
                     audioldm2_audioldm2_00231_seg0000

Output per clip: ``{sample_id}.pdf`` + ``{sample_id}.png``.

Requires: matplotlib.
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (for servers / Colab)
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.models.fusion_model import MultiBranchFusionModel
from src.utils.config import Config
from src.data.dataset import ManifestAudioDataset


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
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    if "spectral" not in model.active_branches:
        raise RuntimeError(
            "This checkpoint was trained with spectral branch disabled. "
            "GradCAM as-written targets spectral_branch.layer3. "
            "Evaluate a full-model or spectral-including checkpoint instead."
        )
    return model


def compute_gradcam(
    model: MultiBranchFusionModel,
    waveform: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run one clip through the model, return (mel_spec_db, cam_heatmap, p_ai).

    - mel_spec_db: (n_mels, T) log-mel spectrogram, dB scale, for plotting.
    - cam_heatmap: (n_mels, T) same resolution as mel_spec_db, in [0, 1].
    - p_ai: scalar sigmoid probability of AI from the regression head.
    """
    waveform = waveform.to(device).unsqueeze(0)  # (1, 1, T)

    # Capture the activation of spectral_branch.layer3.
    activations: dict = {}

    def fwd_hook(_module, _inp, out):
        activations["value"] = out

    handle = model.spectral_branch.layer3.register_forward_hook(fwd_hook)
    try:
        # Model stays in eval mode (for BatchNorm/Dropout) but grad must flow
        # so we can extract the activation's gradient. Use torch.autograd.grad
        # to target the specific activation tensor — avoids populating .grad
        # on every trainable parameter (cleaner than .backward()).
        with torch.enable_grad():
            out = model(waveform)
            score = out["regression_score"].squeeze()  # scalar
            act = activations["value"]                  # (1, C, H, W)
            grad = torch.autograd.grad(outputs=score, inputs=act)[0]
    finally:
        handle.remove()

    # Standard Grad-CAM
    weights = grad.mean(dim=(2, 3), keepdim=True)      # (1, C, 1, 1)
    cam = (weights * act).sum(dim=1, keepdim=True)     # (1, 1, H, W)
    cam = F.relu(cam)
    cam_max = cam.amax(dim=(2, 3), keepdim=True).clamp(min=1e-8)
    cam = cam / cam_max                                 # (1, 1, H, W) in [0, 1]

    # Also compute the mel-spectrogram the spectral branch sees, for display.
    with torch.no_grad():
        mel = model.spectral_branch.mel_spec(waveform.squeeze(1))   # (1, n_mels, T)
        mel_db = model.spectral_branch.amplitude_to_db(mel)         # (1, n_mels, T)

    # Upsample CAM to the mel-spec resolution
    target_size = mel_db.shape[-2:]
    cam_up = F.interpolate(cam, size=target_size, mode="bilinear", align_corners=False)
    cam_np = cam_up[0, 0].detach().cpu().numpy()        # (n_mels, T)
    mel_np = mel_db[0].detach().cpu().numpy()           # (n_mels, T)
    return mel_np, cam_np, float(score.detach().cpu())


def plot_gradcam(
    mel_db: np.ndarray,
    cam: np.ndarray,
    p_ai: float,
    ground_truth: str,
    source: str,
    domain: str,
    sample_id: str,
    duration_s: float,
    out_pdf: Path,
    out_png: Path,
) -> None:
    """Render the 2-panel figure: spectrogram on top, spec+CAM overlay below."""
    n_mels, n_frames = mel_db.shape
    extent = [0.0, duration_s, 0.0, n_mels]  # time (s), mel bin index

    fig, axes = plt.subplots(
        nrows=2, ncols=1, figsize=(5.2, 3.8),
        sharex=True, constrained_layout=True,
    )

    # Panel 1: log-mel spectrogram
    axes[0].imshow(mel_db, origin="lower", aspect="auto",
                   extent=extent, cmap="magma")
    axes[0].set_ylabel("Mel bin")
    axes[0].set_title(
        f"{sample_id}  ({domain}/{source},  GT={ground_truth},  P(AI)={p_ai:.3f})",
        fontsize=9,
    )

    # Panel 2: spectrogram + GradCAM overlay
    axes[1].imshow(mel_db, origin="lower", aspect="auto",
                   extent=extent, cmap="gray")
    axes[1].imshow(cam, origin="lower", aspect="auto",
                   extent=extent, cmap="jet", alpha=0.5,
                   vmin=0.0, vmax=1.0)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Mel bin")

    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def pick_representative_samples(
    manifest_path: str,
    n_per_domain: int = 1,
) -> list[dict]:
    """Select representative test-split clips for the paper figure.

    Returns one AI clip per domain (voice, music, non_human), preferring
    the most visually distinct sources: asvspoof2019 for voice, suno for
    music, audioldm2 for non-human. Falls back to any AI clip if none match.
    """
    preferred = {
        "voice":     ["asvspoof2019", "asvspoof2021"],
        "music":     ["suno", "udio", "musicgen"],
        "non_human": ["audioldm2", "audiogen"],
    }
    picked: list[dict] = []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test"]

    for domain, preferred_sources in preferred.items():
        candidates = [r for r in rows
                      if r["domain"] == domain and float(r["label"]) == 1.0]
        if not candidates:
            continue
        # Prefer the first preferred source that has candidates
        for src in preferred_sources:
            hits = [r for r in candidates if r["source_dataset"] == src]
            if hits:
                picked.extend(hits[:n_per_domain])
                break
        else:
            picked.extend(candidates[:n_per_domain])
    return picked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--output-dir", type=str, default="outputs/figures/gradcam")
    parser.add_argument("--manifest",   type=str, default=None,
                        help="Override config manifest (defaults to segmented manifest).")
    parser.add_argument("--sample-ids", nargs="+", default=None,
                        help="Explicit sample_id values from the manifest. "
                             "Mutually exclusive with --auto-pick.")
    parser.add_argument("--auto-pick",  action="store_true",
                        help="Auto-pick one representative AI clip per domain.")
    args = parser.parse_args()

    if not (args.auto_pick or args.sample_ids):
        parser.error("Pass either --auto-pick or --sample-ids.")

    config = Config.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.manifest or config.data.manifest_path

    # Resolve which sample_ids to process.
    if args.auto_pick:
        records = pick_representative_samples(manifest_path)
        print(f"Auto-picked {len(records)} samples: "
              f"{[r['sample_id'] for r in records]}")
    else:
        with open(manifest_path, newline="", encoding="utf-8") as f:
            by_id = {r["sample_id"]: r for r in csv.DictReader(f)}
        missing = [sid for sid in args.sample_ids if sid not in by_id]
        if missing:
            raise SystemExit(f"Sample IDs not found in manifest: {missing}")
        records = [by_id[sid] for sid in args.sample_ids]

    # Instantiate the Dataset so audio preprocessing (resample/pad) matches
    # training exactly. We only call its _load_audio method; the split filter
    # here is only used to avoid reading the whole manifest twice.
    dataset = ManifestAudioDataset(
        manifest_path=manifest_path,
        data_root=config.data.data_root,
        split="test",
        target_sr=config.audio.sample_rate,
        segment_length=config.data.segment_length,
    )
    _load = dataset._load_audio
    data_root = Path(config.data.data_root)

    model = load_model(config, args.checkpoint, device)
    print(f"Model: active_branches={model.active_branches}")

    duration_s = config.data.segment_length / config.audio.sample_rate

    for rec in records:
        sample_id = rec["sample_id"]
        file_path = str(data_root / rec["file_path"])
        label = float(rec.get("label", 0.0))
        gt = "real" if label < 0.2 else ("mixed" if label < 0.8 else "AI")
        print(f"\n→ {sample_id}  (GT={gt}, source={rec['source_dataset']})")

        waveform = _load(file_path)  # (1, segment_length), mono 16 kHz
        mel_db, cam, p_ai = compute_gradcam(model, waveform, device)

        out_pdf = out_dir / f"{sample_id}.pdf"
        out_png = out_dir / f"{sample_id}.png"
        plot_gradcam(
            mel_db=mel_db, cam=cam, p_ai=p_ai, ground_truth=gt,
            source=rec["source_dataset"], domain=rec["domain"],
            sample_id=sample_id, duration_s=duration_s,
            out_pdf=out_pdf, out_png=out_png,
        )
        print(f"  P(AI)={p_ai:.4f}  wrote {out_pdf.name} + {out_png.name}")

    print(f"\n✓ All figures written to {out_dir}")
    print(f"  To include in paper.tex, add:")
    print(f"    \\includegraphics[width=\\linewidth]{{figures/gradcam/{records[0]['sample_id']}.pdf}}")


if __name__ == "__main__":
    main()

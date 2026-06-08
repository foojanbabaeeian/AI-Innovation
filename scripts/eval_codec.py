"""
Codec robustness evaluation (paper §5.4).

Re-encodes the test split through each codec condition at eval time and
reports EER per condition. Does not require retraining.

Usage:
    python scripts/eval_codec.py \\
        --config configs/gpu_local.yaml \\
        --checkpoint outputs/ai_audio_detection/checkpoint_best.pt \\
        --output-dir outputs/eval/codec_sweep

Requires ffmpeg for MP3/AAC/Opus conditions (VoIP works without ffmpeg).
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.augmentation import CodecAugmentor
from src.data.dataset import ManifestAudioDataset
from src.utils.config import Config

# Reuse evaluate.py helpers without duplicating model/metrics code.
_eval_spec = importlib.util.spec_from_file_location(
    "evaluate_mod", ROOT / "scripts" / "evaluate.py"
)
_eval = importlib.util.module_from_spec(_eval_spec)
_eval_spec.loader.exec_module(_eval)

CODEC_CONDITIONS = [
    ("clean", None),
    ("mp3", 32),
    ("mp3", 64),
    ("mp3", 128),
    ("aac", 32),
    ("aac", 64),
    ("opus", 6),
    ("opus", 12),
    ("opus", 24),
    ("voip", None),
]


def condition_label(codec: str, bitrate: int | None) -> str:
    if codec == "clean":
        return "clean"
    if codec == "voip":
        return "voip_ulaw_8khz"
    return f"{codec}_{bitrate}kbps"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/eval/codec_sweep")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap test set for smoke testing.")
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    augmentor = CodecAugmentor(
        sample_rate=config.audio.sample_rate,
        enabled=True,
        probability=1.0,
    )

    model = _eval.load_model(config, args.checkpoint, device)
    batch_size = args.batch_size or config.data.batch_size
    manifest = args.manifest or config.data.manifest_path

    results = {}
    for codec, bitrate in CODEC_CONDITIONS:
        label = condition_label(codec, bitrate)
        print(f"\n=== Codec condition: {label} ===")

        if codec in ("mp3", "aac", "opus") and not augmentor.has_ffmpeg:
            print(f"  SKIP (ffmpeg not found)")
            results[label] = {"note": "ffmpeg not installed"}
            continue

        def make_transform(c=codec, b=bitrate):
            return lambda w: augmentor.apply_codec(w, c, b)

        dataset = ManifestAudioDataset(
            manifest_path=manifest,
            data_root=config.data.data_root,
            split="test",
            target_sr=config.audio.sample_rate,
            segment_length=config.data.segment_length,
            max_samples=args.max_samples,
            augmentor=make_transform(),
        )
        print(f"  Test set: {len(dataset)} segments")

        pred = _eval.run_inference(
            model, dataset, batch_size, config.data.num_workers, device
        )
        metrics = _eval.aggregate(pred)
        results[label] = metrics
        eer = metrics["overall"].get("binary/eer")
        acc = metrics["overall"].get("classification/accuracy")
        if eer is not None:
            print(f"  EER={eer * 100:.2f}%  acc={acc * 100:.2f}%")

    (out_dir / "codec_metrics.json").write_text(
        json.dumps(results, indent=2, default=float)
    )
    print(f"\n✓ Wrote {out_dir / 'codec_metrics.json'}")


if __name__ == "__main__":
    main()

"""
Training entry point.

Local:
    python -m src.training.train --config configs/default.yaml

Kubernetes:
    Launched via k8s/training-job.yaml

Resume from checkpoint:
    python -m src.training.train --config configs/default.yaml --resume outputs/.../checkpoint_latest.pt

Ablation examples (paper Table 2):
    # single-branch baselines
    python -m src.training.train --config configs/gpu_local.yaml \\
        --disable-branch ssl --disable-branch rawnet \\
        --experiment-name ablation_spectral_only
    python -m src.training.train --config configs/gpu_local.yaml \\
        --disable-branch spectral --disable-branch rawnet \\
        --experiment-name ablation_ssl_only
    python -m src.training.train --config configs/gpu_local.yaml \\
        --disable-branch spectral --disable-branch ssl \\
        --experiment-name ablation_rawnet_only
    # fusion-method baselines
    python -m src.training.train --config configs/gpu_local.yaml \\
        --fusion-method concat  --experiment-name ablation_concat_fusion
    python -m src.training.train --config configs/gpu_local.yaml \\
        --fusion-method average --experiment-name ablation_average_fusion
    # codec-robustness ablation
    python -m src.training.train --config configs/gpu_local.yaml \\
        --no-augmentation --experiment-name ablation_no_codec_aug
"""

import argparse

from src.utils.config import Config
from src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    # Optional overrides (useful for Colab GPU or local training without editing YAML)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    # Ablation knobs: compose with the same YAML config instead of writing one file per ablation.
    parser.add_argument(
        "--disable-branch", action="append", default=None,
        choices=["spectral", "ssl", "rawnet"],
        help="Disable this branch entirely. Repeatable: "
             "`--disable-branch ssl --disable-branch rawnet` yields a spectral-only model.",
    )
    parser.add_argument(
        "--fusion-method", type=str, default=None,
        choices=["attention", "concat", "average"],
        help="Override the fusion strategy. 'attention' is the paper default; "
             "'concat' and 'average' are ablation baselines.",
    )
    parser.add_argument(
        "--experiment-name", type=str, default=None,
        help="Override logging.experiment_name so ablation outputs land in a separate folder.",
    )
    parser.add_argument(
        "--no-augmentation", action="store_true",
        help="Disable codec augmentation for this run (used for the no-aug ablation row).",
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config)

    # Apply CLI overrides if provided
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.num_workers is not None:
        config.data.num_workers = args.num_workers
    if args.gradient_accumulation_steps is not None:
        config.training.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.disable_branch:
        config.model.disable_branches = list(args.disable_branch)
    if args.fusion_method is not None:
        config.model.fusion_method = args.fusion_method
    if args.experiment_name is not None:
        config.logging.experiment_name = args.experiment_name
    if args.no_augmentation:
        config.augmentation.enabled = False

    trainer = Trainer(config)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.fit()


if __name__ == "__main__":
    main()

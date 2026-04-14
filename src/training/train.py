"""
Training entry point.

Local:
    python -m src.training.train --config configs/default.yaml

Kubernetes:
    Launched via k8s/training-job.yaml

Resume from checkpoint:
    python -m src.training.train --config configs/default.yaml --resume outputs/.../checkpoint_latest.pt
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
    args = parser.parse_args()

    config = Config.from_yaml(args.config)

    # Apply CLI overrides if provided
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.num_workers is not None:
        config.data.num_workers = args.num_workers
    if args.gradient_accumulation_steps is not None:
        config.training.gradient_accumulation_steps = args.gradient_accumulation_steps

    trainer = Trainer(config)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.fit()


if __name__ == "__main__":
    main()

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
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    trainer = Trainer(config)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.fit()


if __name__ == "__main__":
    main()

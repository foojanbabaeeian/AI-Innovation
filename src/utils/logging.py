"""
Logging utilities for training: console logging + Weights & Biases integration.

Provides a unified interface so the training loop doesn't need to know
whether W&B is enabled or not.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from src.config import Config, LoggingConfig


def setup_logger(
    name: str = "audio_detection",
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Set up a console (and optional file) logger.

    Args:
        name: Logger name.
        level: Logging level string ('DEBUG', 'INFO', 'WARNING', etc.).
        log_file: Optional path to also write logs to a file.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s — %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # Optional file handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class WandbLogger:
    """Wrapper around Weights & Biases for experiment tracking.

    Gracefully handles the case where W&B is disabled or not installed.

    Args:
        config: Full system Config (logged as hyperparameters).
        enabled: Whether W&B logging is active.
        project: W&B project name.
        entity: W&B entity (username or team).
        tags: List of tags for the run.
        run_name: Optional custom run name.
    """

    def __init__(
        self,
        config: Config,
        enabled: bool = True,
        project: str = "ai-audio-detection",
        entity: Optional[str] = None,
        tags: Optional[list] = None,
        run_name: Optional[str] = None,
    ):
        self.enabled = enabled
        self.run = None

        if not enabled:
            return

        try:
            import wandb

            self.wandb = wandb

            # Convert config dataclass to dict for W&B
            config_dict = self._dataclass_to_dict(config)

            self.run = wandb.init(
                project=project,
                entity=entity,
                tags=tags or [],
                name=run_name,
                config=config_dict,
                reinit=True,
            )
        except ImportError:
            print("WARNING: wandb not installed. Disabling W&B logging.")
            print("  Install with: pip install wandb")
            self.enabled = False
        except Exception as e:
            print(f"WARNING: Could not initialize W&B: {e}")
            print("  Run 'wandb login' to authenticate, or set enabled=False.")
            self.enabled = False

    def _dataclass_to_dict(self, obj) -> dict:
        """Recursively convert a dataclass to a plain dict.

        Args:
            obj: Dataclass instance or primitive value.

        Returns:
            Nested dictionary representation.
        """
        if hasattr(obj, "__dataclass_fields__"):
            return {k: self._dataclass_to_dict(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, list):
            return [self._dataclass_to_dict(v) for v in obj]
        return obj

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to W&B.

        Args:
            metrics: Dictionary of metric name → value.
            step: Optional global step number.
        """
        if not self.enabled or self.run is None:
            return
        self.wandb.log(metrics, step=step)

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        eer: Optional[float] = None,
        min_dcf: Optional[float] = None,
        lr: Optional[float] = None,
        extra: Optional[dict] = None,
    ):
        """Log end-of-epoch summary metrics.

        Args:
            epoch: Current epoch number.
            train_loss: Training loss for the epoch.
            val_loss: Validation loss.
            eer: Equal Error Rate on validation set.
            min_dcf: Minimum Detection Cost Function.
            lr: Current learning rate.
            extra: Additional metrics to log.
        """
        metrics = {"epoch": epoch, "train/loss": train_loss}

        if val_loss is not None:
            metrics["val/loss"] = val_loss
        if eer is not None:
            metrics["val/eer"] = eer
        if min_dcf is not None:
            metrics["val/min_dcf"] = min_dcf
        if lr is not None:
            metrics["train/lr"] = lr
        if extra:
            metrics.update(extra)

        self.log(metrics, step=epoch)

    def log_audio(
        self,
        key: str,
        audio_tensor,
        sample_rate: int = 16000,
        caption: Optional[str] = None,
    ):
        """Log an audio sample to W&B.

        Args:
            key: Metric key for the audio.
            audio_tensor: 1D tensor or numpy array of audio samples.
            sample_rate: Audio sample rate.
            caption: Optional caption.
        """
        if not self.enabled or self.run is None:
            return
        import numpy as np

        if hasattr(audio_tensor, "numpy"):
            audio_tensor = audio_tensor.numpy()
        audio_tensor = np.squeeze(audio_tensor)

        self.wandb.log({
            key: self.wandb.Audio(audio_tensor, sample_rate=sample_rate, caption=caption)
        })

    def log_image(self, key: str, image, caption: Optional[str] = None):
        """Log an image (e.g., spectrogram heatmap) to W&B.

        Args:
            key: Metric key for the image.
            image: PIL Image, numpy array, or matplotlib figure.
            caption: Optional caption.
        """
        if not self.enabled or self.run is None:
            return
        self.wandb.log({key: self.wandb.Image(image, caption=caption)})

    def finish(self):
        """Finish the W&B run."""
        if self.enabled and self.run is not None:
            self.run.finish()

    @staticmethod
    def from_config(config: Config) -> "WandbLogger":
        """Create WandbLogger from Config.

        Args:
            config: Full system Config.

        Returns:
            WandbLogger instance.
        """
        wcfg = config.logging.wandb
        return WandbLogger(
            config=config,
            enabled=wcfg.enabled,
            project=wcfg.project,
            entity=wcfg.entity,
            tags=wcfg.tags,
        )

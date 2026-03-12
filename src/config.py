"""
Configuration management using dataclasses + YAML.

Loads configs/default.yaml and merges with any overrides.
Provides typed access to all hyperparameters.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class MelConfig:
    """Mel-spectrogram extraction parameters."""
    n_mels: int = 128
    n_fft: int = 1024
    hop_length: int = 256
    f_min: int = 0
    f_max: int = 8000
    power: float = 2.0
    normalize: bool = True


@dataclass
class AudioConfig:
    """Audio preprocessing parameters."""
    sample_rate: int = 16000
    duration_sec: float = 4.0
    n_samples: int = 64000
    mel: MelConfig = field(default_factory=MelConfig)


@dataclass
class CodecConfig:
    """Codec augmentation parameters."""
    mp3_bitrates: List[int] = field(default_factory=lambda: [32, 64, 128])
    aac_bitrates: List[int] = field(default_factory=lambda: [32, 64])
    opus_bitrates: List[int] = field(default_factory=lambda: [6, 12, 24])
    voip_sample_rate: int = 8000


@dataclass
class AugmentationConfig:
    """Data augmentation parameters."""
    enabled: bool = True
    probability: float = 0.5
    codec: CodecConfig = field(default_factory=CodecConfig)


@dataclass
class DataLoaderConfig:
    """DataLoader parameters."""
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2


@dataclass
class ResNetConfig:
    """ResNet-34 branch config."""
    backbone: str = "resnet34"
    pretrained: bool = True
    feature_dim: int = 512


@dataclass
class WavLMConfig:
    """WavLM branch config."""
    model_name: str = "microsoft/wavlm-base-plus"
    freeze_layers: int = 8
    feature_dim: int = 768


@dataclass
class RawNet2Config:
    """RawNet2 branch config."""
    sinc_channels: int = 128
    feature_dim: int = 128


@dataclass
class FusionConfig:
    """Fusion transformer config."""
    num_heads: int = 4
    num_layers: int = 2
    dim_feedforward: int = 512
    dropout: float = 0.1


@dataclass
class ClassifierConfig:
    """Output classifier config."""
    num_classes: int = 2
    dropout: float = 0.3


@dataclass
class ModelConfig:
    """Full model architecture config."""
    projection_dim: int = 256
    resnet: ResNetConfig = field(default_factory=ResNetConfig)
    wavlm: WavLMConfig = field(default_factory=WavLMConfig)
    rawnet2: RawNet2Config = field(default_factory=RawNet2Config)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)


@dataclass
class OptimizerConfig:
    """Optimizer parameters."""
    name: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 1e-4
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])


@dataclass
class SchedulerConfig:
    """Learning rate scheduler parameters."""
    name: str = "cosine"
    warmup_epochs: int = 5
    min_lr: float = 1e-6


@dataclass
class EarlyStoppingConfig:
    """Early stopping parameters."""
    patience: int = 10
    metric: str = "eer"
    mode: str = "min"


@dataclass
class TrainingConfig:
    """Training loop parameters."""
    epochs: int = 50
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    gradient_clip: float = 1.0
    mixed_precision: bool = True


@dataclass
class CheckpointConfig:
    """Checkpoint saving parameters."""
    save_dir: str = "checkpoints"
    save_every_epoch: bool = True
    keep_top_k: int = 3
    drive_mount: str = "/content/drive/MyDrive/AudioDetection/checkpoints"


@dataclass
class WandbConfig:
    """Weights & Biases config."""
    enabled: bool = True
    project: str = "ai-audio-detection"
    entity: Optional[str] = None
    tags: List[str] = field(default_factory=lambda: ["capstone", "audio-deepfake"])
    log_every_n_steps: int = 50


@dataclass
class LoggingConfig:
    """Logging config."""
    console_level: str = "INFO"
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass
class SmokeTestConfig:
    """Smoke test mode for local debugging."""
    enabled: bool = False
    num_samples: int = 10
    num_batches: int = 2
    batch_size: int = 4


@dataclass
class MinDCFConfig:
    """min-DCF evaluation parameters."""
    p_target: float = 0.05
    c_miss: float = 1.0
    c_fa: float = 1.0


@dataclass
class EvaluationConfig:
    """Evaluation parameters."""
    metrics: List[str] = field(default_factory=lambda: ["eer", "min_dcf"])
    min_dcf: MinDCFConfig = field(default_factory=MinDCFConfig)


@dataclass
class Config:
    """Top-level configuration for the entire system."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    smoke_test: SmokeTestConfig = field(default_factory=SmokeTestConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def _merge_dict_into_dataclass(dc, d):
    """Recursively merge a dictionary into a dataclass instance.

    Args:
        dc: Dataclass instance to update.
        d: Dictionary with override values.

    Returns:
        The updated dataclass instance.
    """
    for key, value in d.items():
        if not hasattr(dc, key):
            continue
        current = getattr(dc, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dict_into_dataclass(current, value)
        else:
            setattr(dc, key, value)
    return dc


def load_config(config_path: Optional[str] = None, overrides: Optional[dict] = None) -> Config:
    """Load configuration from YAML file with optional overrides.

    Args:
        config_path: Path to YAML config file. Defaults to configs/default.yaml.
        overrides: Dictionary of overrides to apply on top of the YAML config.

    Returns:
        Fully populated Config dataclass.
    """
    config = Config()

    if config_path is None:
        # Look for default config relative to project root
        project_root = Path(__file__).parent.parent
        config_path = project_root / "configs" / "default.yaml"

    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path, "r") as f:
            yaml_dict = yaml.safe_load(f) or {}

        # Flatten codec config from YAML structure
        if "augmentation" in yaml_dict and "codecs" in yaml_dict["augmentation"]:
            codecs = yaml_dict["augmentation"].pop("codecs")
            yaml_dict["augmentation"]["codec"] = {
                "mp3_bitrates": codecs.get("mp3", {}).get("bitrates", [32, 64, 128]),
                "aac_bitrates": codecs.get("aac", {}).get("bitrates", [32, 64]),
                "opus_bitrates": codecs.get("opus", {}).get("bitrates", [6, 12, 24]),
                "voip_sample_rate": codecs.get("voip", {}).get("sample_rate", 8000),
            }

        _merge_dict_into_dataclass(config, yaml_dict)

    if overrides:
        _merge_dict_into_dataclass(config, overrides)

    # Ensure n_samples is consistent
    config.audio.n_samples = int(config.audio.sample_rate * config.audio.duration_sec)

    return config

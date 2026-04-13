"""
Training configuration using dataclasses.
Loaded from YAML config files in configs/ directory.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    manifest_path: str = "data/metadata/master_manifest.csv"
    data_root: str = "data"
    target_sr: int = 16000
    segment_length: int = 64000  # 4 seconds @ 16kHz
    batch_size: int = 32
    num_workers: int = 8  # parallel data loading threads
    augment_prob: float = 0.5
    threshold_low: float = 0.2
    threshold_high: float = 0.8


@dataclass
class ModelConfig:
    embed_dim: int = 128
    num_attention_heads: int = 4
    num_attention_layers: int = 2
    num_classes: int = 3  # real, mixed, AI
    ssl_model_name: str = "microsoft/wavlm-base-plus"
    freeze_ssl_feature_extractor: bool = True
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    loss_alpha: float = 0.5  # regression vs classification weight
    grad_clip_norm: float = 1.0
    early_stopping_patience: int = 10
    scheduler: str = "cosine"  # cosine | step | plateau
    mixed_precision: bool = False
    gradient_accumulation_steps: int = 4


@dataclass
class LoggingConfig:
    output_dir: str = "outputs"
    experiment_name: str = "ai_audio_detection"
    log_every_n_steps: int = 50
    eval_every_n_epochs: int = 1
    save_top_k: int = 3
    wandb_project: str = "ai-audio-detection"
    wandb_enabled: bool = False


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f)

        config = cls()
        for section_name, section_cls in [
            ("data", DataConfig),
            ("model", ModelConfig),
            ("training", TrainingConfig),
            ("logging", LoggingConfig),
        ]:
            if section_name in raw:
                setattr(config, section_name, section_cls(**raw[section_name]))
        return config

    def to_yaml(self, path: str):
        from dataclasses import asdict
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

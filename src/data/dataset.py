"""
Manifest-based Dataset and DataLoader for the AI audio detection pipeline.

Reads master_manifest.csv produced by ingest_all.py + run_preprocessing,
and returns preprocessed audio tensors for training and validation.

Expected manifest columns:
    file_path   — audio file path relative to data_root
    label       — float 0.0 (real) to 1.0 (AI-generated)
    split       — "train", "val", or "test"
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


class ManifestAudioDataset(Dataset):
    """PyTorch Dataset backed by a master manifest CSV.

    Each row in the manifest is one audio segment.  The dataset loads,
    resamples, pads/truncates, and returns three tensors needed by the trainer:

        waveform    (1, segment_length)  — raw mono audio
        ai_ratio    scalar float         — same as label (0.0=real, 1.0=AI)
        class_label scalar long          — 0 real | 1 mixed | 2 AI

    Args:
        manifest_path: Path to master_manifest.csv.
        data_root: Root directory; file_path in manifest is relative to this.
        split: One of "train", "val", "test".
        target_sr: Target sample rate in Hz.
        segment_length: Fixed number of samples per segment.
        max_samples: Optional cap for smoke-testing.
    """

    def __init__(
        self,
        manifest_path: str,
        data_root: str,
        split: str,
        target_sr: int = 16000,
        segment_length: int = 64000,
        max_samples: Optional[int] = None,
    ):
        self.data_root = Path(data_root)
        self.target_sr = target_sr
        self.segment_length = segment_length

        df = pd.read_csv(manifest_path)
        df = df[df["split"] == split].reset_index(drop=True)

        if max_samples is not None:
            df = df.iloc[:max_samples]

        self.records = df.to_dict("records")
        logger.info("Split=%s: %d samples from %s", split, len(self.records), manifest_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        row = self.records[idx]
        path = self.data_root / row["file_path"]

        # Support both "label" and "ai_ratio" column names in the manifest
        label = float(row.get("label", row.get("ai_ratio", 0.0)))

        waveform = self._load_audio(str(path))

        return {
            "waveform": waveform,
            "ai_ratio": torch.tensor(label, dtype=torch.float32),
            "class_label": self._label_to_class(label),
        }

    def _load_audio(self, path: str) -> torch.Tensor:
        """Load audio, convert to mono, resample, pad/truncate to segment_length."""
        try:
            wav, sr = torchaudio.load(path)
        except Exception as exc:
            logger.warning("Failed to load %s: %s — returning silence", path, exc)
            return torch.zeros(1, self.segment_length)

        # Mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.target_sr:
            wav = T.Resample(orig_freq=sr, new_freq=self.target_sr)(wav)

        # Pad or truncate to fixed length
        n = wav.shape[1]
        if n >= self.segment_length:
            wav = wav[:, :self.segment_length]
        else:
            wav = torch.nn.functional.pad(wav, (0, self.segment_length - n))

        return wav  # (1, segment_length)

    @staticmethod
    def _label_to_class(label: float) -> torch.Tensor:
        """Map continuous label to 3-class integer label.

        0 = real   (label < 0.2)
        1 = mixed  (0.2 <= label < 0.8)
        2 = AI     (label >= 0.8)
        """
        if label < 0.2:
            cls = 0
        elif label < 0.8:
            cls = 1
        else:
            cls = 2
        return torch.tensor(cls, dtype=torch.long)


def build_dataloader(
    manifest_path: str,
    data_root: str,
    split: str,
    batch_size: int = 32,
    num_workers: int = 4,
    target_sr: int = 16000,
    segment_length: int = 64000,
    max_samples: Optional[int] = None,
) -> Optional[DataLoader]:
    """Build a DataLoader for one split from the master manifest.

    Returns None if the manifest is missing or the split is empty.

    Args:
        manifest_path: Path to master_manifest.csv.
        data_root: Root directory; file_path in manifest is relative to this.
        split: "train", "val", or "test".
        batch_size: Number of samples per batch.
        num_workers: Parallel data loading workers.
        target_sr: Target sample rate in Hz.
        segment_length: Fixed number of samples per segment.
        max_samples: Cap dataset size for smoke-testing.

    Returns:
        DataLoader or None.
    """
    if not Path(manifest_path).exists():
        logger.warning("Manifest not found: %s — returning None", manifest_path)
        return None

    dataset = ManifestAudioDataset(
        manifest_path=manifest_path,
        data_root=data_root,
        split=split,
        target_sr=target_sr,
        segment_length=segment_length,
        max_samples=max_samples,
    )

    if len(dataset) == 0:
        logger.warning("No samples for split=%s in %s", split, manifest_path)
        return None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == "train"),
    )

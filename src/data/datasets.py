"""
PyTorch Dataset classes for all audio detection datasets.

Each dataset class handles its own file discovery and label parsing.
All datasets return the same output format for compatibility with the
unified training pipeline.

Output format per sample:
{
    'mel_spectrogram': Tensor (1, n_mels, T),
    'waveform_wavlm': Tensor (n_samples,),
    'waveform_rawnet2': Tensor (1, n_samples),
    'label': int (0 = human/real, 1 = AI-generated/spoof),
    'metadata': dict with dataset-specific info
}
"""

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import ConcatDataset, Dataset

from src.data.augmentation import CodecAugmentor
from src.data.preprocessing import AudioPreprocessor


class ASVspoofDataset(Dataset):
    """ASVspoof 2019 LA dataset.

    Parses the official CM protocol files to get utterance IDs and labels.
    Labels: 'bonafide' → 0 (real), 'spoof' → 1 (fake).

    Args:
        root_dir: Root directory of ASVspoof2019_LA.
        partition: One of 'train', 'dev', 'eval'.
        preprocessor: AudioPreprocessor instance.
        augmentor: Optional CodecAugmentor for training-time augmentation.
        max_samples: Limit number of samples (for smoke testing).
    """

    def __init__(
        self,
        root_dir: str,
        partition: str = "train",
        preprocessor: AudioPreprocessor = None,
        augmentor: Optional[CodecAugmentor] = None,
        max_samples: Optional[int] = None,
    ):
        self.root_dir = Path(root_dir)
        self.partition = partition
        self.preprocessor = preprocessor
        self.augmentor = augmentor

        # Map partition names to protocol/audio directory names
        partition_map = {
            "train": ("ASVspoof2019.LA.cm.train.trn.txt", "ASVspoof2019_LA_train"),
            "dev": ("ASVspoof2019.LA.cm.dev.trl.txt", "ASVspoof2019_LA_dev"),
            "eval": ("ASVspoof2019.LA.cm.eval.trl.txt", "ASVspoof2019_LA_eval"),
        }

        if partition not in partition_map:
            raise ValueError(f"Partition must be one of {list(partition_map.keys())}")

        protocol_file, audio_subdir = partition_map[partition]
        self.protocol_path = (
            self.root_dir / "ASVspoof2019_LA_cm_protocols" / protocol_file
        )
        self.audio_dir = self.root_dir / audio_subdir / "flac"

        # Parse protocol file
        self.samples = self._parse_protocol()

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def _parse_protocol(self) -> List[Dict]:
        """Parse ASVspoof protocol file to extract utterance info.

        Protocol format: SPEAKER_ID UTTERANCE_ID - SYSTEM_ID LABEL
        Example: LA_0079 LA_T_1138215 - A07 spoof

        Returns:
            List of dicts with keys: 'utt_id', 'speaker_id', 'system_id', 'label'.
        """
        samples = []

        if not self.protocol_path.exists():
            print(f"WARNING: Protocol file not found: {self.protocol_path}")
            return samples

        with open(self.protocol_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                speaker_id = parts[0]
                utt_id = parts[1]
                system_id = parts[3]
                label_str = parts[4]
                label = 0 if label_str == "bonafide" else 1

                audio_path = self.audio_dir / f"{utt_id}.flac"
                samples.append({
                    "utt_id": utt_id,
                    "speaker_id": speaker_id,
                    "system_id": system_id,
                    "label": label,
                    "audio_path": str(audio_path),
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """Load and preprocess a single sample.

        Args:
            idx: Sample index.

        Returns:
            Dict with mel_spectrogram, waveform_wavlm, waveform_rawnet2,
            label, and metadata.
        """
        sample = self.samples[idx]
        audio_path = sample["audio_path"]

        # Load and preprocess
        waveform = self.preprocessor.load_audio(audio_path)

        # Apply augmentation if available
        if self.augmentor is not None:
            waveform = self.augmentor(waveform)

        return {
            "mel_spectrogram": self.preprocessor.compute_mel_spectrogram(waveform),
            "waveform_wavlm": self.preprocessor.get_raw_waveform(waveform),
            "waveform_rawnet2": self.preprocessor.get_rawnet2_input(waveform),
            "label": sample["label"],
            "metadata": {
                "dataset": "asvspoof2019",
                "partition": self.partition,
                "utt_id": sample["utt_id"],
                "system_id": sample["system_id"],
            },
        }


class ASVspoof2021Dataset(Dataset):
    """ASVspoof 2021 LA evaluation dataset.

    Similar to 2019 but only has an eval partition with a different protocol format.

    Args:
        root_dir: Root directory of ASVspoof2021_LA.
        preprocessor: AudioPreprocessor instance.
        augmentor: Optional CodecAugmentor.
        max_samples: Limit number of samples (for smoke testing).
    """

    def __init__(
        self,
        root_dir: str,
        preprocessor: AudioPreprocessor = None,
        augmentor: Optional[CodecAugmentor] = None,
        max_samples: Optional[int] = None,
    ):
        self.root_dir = Path(root_dir)
        self.preprocessor = preprocessor
        self.augmentor = augmentor

        self.protocol_path = (
            self.root_dir / "ASVspoof2021_LA_cm_protocols"
            / "ASVspoof2021.LA.cm.eval.trl.txt"
        )
        self.audio_dir = self.root_dir / "flac"

        self.samples = self._parse_protocol()

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def _parse_protocol(self) -> List[Dict]:
        """Parse ASVspoof 2021 protocol file.

        Format: SPEAKER_ID UTTERANCE_ID - SYSTEM_ID LABEL
        (same as 2019 but eval-only)

        Returns:
            List of sample dicts.
        """
        samples = []

        if not self.protocol_path.exists():
            print(f"WARNING: Protocol file not found: {self.protocol_path}")
            return samples

        with open(self.protocol_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                speaker_id = parts[0]
                utt_id = parts[1]
                system_id = parts[3]
                label_str = parts[4]
                label = 0 if label_str == "bonafide" else 1

                audio_path = self.audio_dir / f"{utt_id}.flac"
                samples.append({
                    "utt_id": utt_id,
                    "speaker_id": speaker_id,
                    "system_id": system_id,
                    "label": label,
                    "audio_path": str(audio_path),
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        waveform = self.preprocessor.load_audio(sample["audio_path"])

        if self.augmentor is not None:
            waveform = self.augmentor(waveform)

        return {
            "mel_spectrogram": self.preprocessor.compute_mel_spectrogram(waveform),
            "waveform_wavlm": self.preprocessor.get_raw_waveform(waveform),
            "waveform_rawnet2": self.preprocessor.get_rawnet2_input(waveform),
            "label": sample["label"],
            "metadata": {
                "dataset": "asvspoof2021",
                "utt_id": sample["utt_id"],
                "system_id": sample["system_id"],
            },
        }


class WaveFakeDataset(Dataset):
    """WaveFake dataset (Joel Frank et al., 2021).

    Directory structure: WaveFake/{generator_name}/*.wav for fake,
    WaveFake/real/*.wav for real. Labels derived from directory names.

    Args:
        root_dir: Root directory of WaveFake.
        preprocessor: AudioPreprocessor instance.
        augmentor: Optional CodecAugmentor.
        max_samples: Limit number of samples (for smoke testing).
    """

    def __init__(
        self,
        root_dir: str,
        preprocessor: AudioPreprocessor = None,
        augmentor: Optional[CodecAugmentor] = None,
        max_samples: Optional[int] = None,
    ):
        self.root_dir = Path(root_dir)
        self.preprocessor = preprocessor
        self.augmentor = augmentor

        self.samples = self._discover_samples()

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def _discover_samples(self) -> List[Dict]:
        """Walk the WaveFake directory tree and assign labels.

        Directories containing 'real' or 'bonafide' → label 0.
        All other directories → label 1 (AI-generated).

        Returns:
            List of sample dicts.
        """
        samples = []

        if not self.root_dir.exists():
            print(f"WARNING: WaveFake directory not found: {self.root_dir}")
            return samples

        for subdir in sorted(self.root_dir.iterdir()):
            if not subdir.is_dir():
                continue

            dirname_lower = subdir.name.lower()
            is_real = "real" in dirname_lower or "bonafide" in dirname_lower
            label = 0 if is_real else 1
            generator = "human" if is_real else subdir.name

            for audio_file in sorted(subdir.glob("*.wav")):
                samples.append({
                    "audio_path": str(audio_file),
                    "label": label,
                    "generator": generator,
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        waveform = self.preprocessor.load_audio(sample["audio_path"])

        if self.augmentor is not None:
            waveform = self.augmentor(waveform)

        return {
            "mel_spectrogram": self.preprocessor.compute_mel_spectrogram(waveform),
            "waveform_wavlm": self.preprocessor.get_raw_waveform(waveform),
            "waveform_rawnet2": self.preprocessor.get_rawnet2_input(waveform),
            "label": sample["label"],
            "metadata": {
                "dataset": "wavefake",
                "generator": sample["generator"],
            },
        }


class MusicAIDataset(Dataset):
    """Dataset for music/non-human audio detection.

    Combines:
    - Real music from MusicCaps (label 0)
    - AI-generated music from Suno/Udio (label 1)

    Args:
        real_music_dir: Directory with real music WAV files (MusicCaps audio).
        ai_music_dir: Directory with AI-generated music WAV files.
        preprocessor: AudioPreprocessor instance.
        augmentor: Optional CodecAugmentor.
        max_samples: Limit number of samples (for smoke testing).
    """

    def __init__(
        self,
        real_music_dir: str,
        ai_music_dir: str,
        preprocessor: AudioPreprocessor = None,
        augmentor: Optional[CodecAugmentor] = None,
        max_samples: Optional[int] = None,
    ):
        self.preprocessor = preprocessor
        self.augmentor = augmentor

        self.samples = self._discover_samples(
            Path(real_music_dir), Path(ai_music_dir)
        )

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def _discover_samples(
        self, real_dir: Path, ai_dir: Path
    ) -> List[Dict]:
        """Find all audio files in real and AI music directories.

        Args:
            real_dir: Path to real music audio files.
            ai_dir: Path to AI-generated music audio files.

        Returns:
            List of sample dicts.
        """
        samples = []
        audio_extensions = {"*.wav", "*.flac", "*.mp3", "*.ogg"}

        for ext in audio_extensions:
            if real_dir.exists():
                for f in sorted(real_dir.rglob(ext)):
                    samples.append({
                        "audio_path": str(f),
                        "label": 0,
                        "source": "real_music",
                    })
            if ai_dir.exists():
                for f in sorted(ai_dir.rglob(ext)):
                    samples.append({
                        "audio_path": str(f),
                        "label": 1,
                        "source": "ai_music",
                    })

        if not samples:
            print(
                f"WARNING: No audio files found in {real_dir} or {ai_dir}. "
                "See src/data/download.py for instructions."
            )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        waveform = self.preprocessor.load_audio(sample["audio_path"])

        if self.augmentor is not None:
            waveform = self.augmentor(waveform)

        return {
            "mel_spectrogram": self.preprocessor.compute_mel_spectrogram(waveform),
            "waveform_wavlm": self.preprocessor.get_raw_waveform(waveform),
            "waveform_rawnet2": self.preprocessor.get_rawnet2_input(waveform),
            "label": sample["label"],
            "metadata": {
                "dataset": "music_ai",
                "source": sample["source"],
            },
        }


class UnifiedAudioDataset(Dataset):
    """Unified dataset that combines all individual datasets for training.

    Wraps multiple datasets into a single interface with consistent output format.
    Useful for cross-domain training where we want to mix voice and music samples.

    Args:
        datasets: List of individual dataset instances.
        shuffle_seed: Random seed for shuffling the combined index. If None, no shuffle.
    """

    def __init__(
        self,
        datasets: List[Dataset],
        shuffle_seed: Optional[int] = None,
    ):
        self.concat_dataset = ConcatDataset(datasets)
        self.indices = list(range(len(self.concat_dataset)))

        if shuffle_seed is not None:
            import random as rng
            rng_instance = rng.Random(shuffle_seed)
            rng_instance.shuffle(self.indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        return self.concat_dataset[self.indices[idx]]


def build_dataloaders(config) -> dict:
    """Build train/dev/eval DataLoaders from config.

    Creates all dataset instances, wraps them in DataLoaders, and returns
    them as a dictionary. Handles smoke test mode by limiting samples.

    Args:
        config: Full system Config.

    Returns:
        Dict with keys 'train', 'dev', 'eval', each mapping to a DataLoader.
    """
    from torch.utils.data import DataLoader

    preprocessor = AudioPreprocessor.from_config(config)

    # Create augmentor for training only
    train_augmentor = CodecAugmentor.from_config(config)
    eval_augmentor = None  # No augmentation during evaluation

    max_samples = config.smoke_test.num_samples if config.smoke_test.enabled else None
    batch_size = (
        config.smoke_test.batch_size
        if config.smoke_test.enabled
        else config.dataloader.batch_size
    )

    # --- Training datasets ---
    train_datasets = []

    # ASVspoof 2019 train
    asvspoof_train = ASVspoofDataset(
        root_dir="data/ASVspoof2019_LA",
        partition="train",
        preprocessor=preprocessor,
        augmentor=train_augmentor,
        max_samples=max_samples,
    )
    if len(asvspoof_train) > 0:
        train_datasets.append(asvspoof_train)

    # WaveFake
    wavefake = WaveFakeDataset(
        root_dir="data/WaveFake",
        preprocessor=preprocessor,
        augmentor=train_augmentor,
        max_samples=max_samples,
    )
    if len(wavefake) > 0:
        train_datasets.append(wavefake)

    # Music AI
    music_ai = MusicAIDataset(
        real_music_dir="data/MusicCaps/audio",
        ai_music_dir="data/AI_Music",
        preprocessor=preprocessor,
        augmentor=train_augmentor,
        max_samples=max_samples,
    )
    if len(music_ai) > 0:
        train_datasets.append(music_ai)

    # --- Dev dataset ---
    dev_datasets = []
    asvspoof_dev = ASVspoofDataset(
        root_dir="data/ASVspoof2019_LA",
        partition="dev",
        preprocessor=preprocessor,
        augmentor=eval_augmentor,
        max_samples=max_samples,
    )
    if len(asvspoof_dev) > 0:
        dev_datasets.append(asvspoof_dev)

    # --- Eval dataset ---
    eval_datasets = []
    asvspoof_eval = ASVspoofDataset(
        root_dir="data/ASVspoof2019_LA",
        partition="eval",
        preprocessor=preprocessor,
        augmentor=eval_augmentor,
        max_samples=max_samples,
    )
    if len(asvspoof_eval) > 0:
        eval_datasets.append(asvspoof_eval)

    # ASVspoof 2021
    asvspoof2021 = ASVspoof2021Dataset(
        root_dir="data/ASVspoof2021_LA",
        preprocessor=preprocessor,
        augmentor=eval_augmentor,
        max_samples=max_samples,
    )
    if len(asvspoof2021) > 0:
        eval_datasets.append(asvspoof2021)

    def _collate_fn(batch: List[dict]) -> dict:
        """Custom collate function to handle our dict-based samples.

        Args:
            batch: List of sample dicts from __getitem__.

        Returns:
            Batched dict with stacked tensors and list of metadata dicts.
        """
        return {
            "mel_spectrogram": torch.stack([s["mel_spectrogram"] for s in batch]),
            "waveform_wavlm": torch.stack([s["waveform_wavlm"] for s in batch]),
            "waveform_rawnet2": torch.stack([s["waveform_rawnet2"] for s in batch]),
            "label": torch.tensor([s["label"] for s in batch], dtype=torch.long),
            "metadata": [s["metadata"] for s in batch],
        }

    num_workers = 0 if config.smoke_test.enabled else config.dataloader.num_workers

    loaders = {}

    if train_datasets:
        train_unified = UnifiedAudioDataset(train_datasets, shuffle_seed=42)
        loaders["train"] = DataLoader(
            train_unified,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=config.dataloader.pin_memory and torch.cuda.is_available(),
            collate_fn=_collate_fn,
            drop_last=True,
        )
    else:
        loaders["train"] = None

    if dev_datasets:
        dev_unified = UnifiedAudioDataset(dev_datasets)
        loaders["dev"] = DataLoader(
            dev_unified,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=config.dataloader.pin_memory and torch.cuda.is_available(),
            collate_fn=_collate_fn,
        )
    else:
        loaders["dev"] = None

    if eval_datasets:
        eval_unified = UnifiedAudioDataset(eval_datasets)
        loaders["eval"] = DataLoader(
            eval_unified,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=config.dataloader.pin_memory and torch.cuda.is_available(),
            collate_fn=_collate_fn,
        )
    else:
        loaders["eval"] = None

    return loaders

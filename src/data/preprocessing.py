"""
Audio preprocessing pipeline.

Handles three input representations needed by the model branches:
1. Mel-spectrogram (128 bins × T frames) — for ResNet-34
2. Raw waveform at 16kHz — for WavLM
3. Fixed-length raw waveform (64k samples) — for RawNet2

All preprocessing is deterministic (no randomness) so that evaluation
is reproducible. Augmentation is handled separately in augmentation.py.
"""

import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T

from src.utils.config import AudioConfig, MelConfig


class AudioPreprocessor:
    """Loads audio files and produces all three input representations.

    Args:
        config: AudioConfig with sample rate, duration, and mel parameters.
    """

    def __init__(self, config: AudioConfig):
        self.sample_rate = config.sample_rate
        self.n_samples = config.n_samples
        self.mel_config = config.mel

        # Build the Mel-spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.mel_config.n_fft,
            hop_length=self.mel_config.hop_length,
            n_mels=self.mel_config.n_mels,
            f_min=self.mel_config.f_min,
            f_max=self.mel_config.f_max,
            power=self.mel_config.power,
        )
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80)

    def load_audio(self, file_path: str) -> torch.Tensor:
        """Load an audio file and resample to target sample rate.

        Args:
            file_path: Path to audio file (WAV, FLAC, MP3, etc.).

        Returns:
            Mono waveform tensor of shape (1, n_samples), zero-padded or
            truncated to exactly n_samples.
        """
        data, sr = sf.read(file_path, dtype="float32")
        # sf.read returns (n_samples,) for mono, (n_samples, channels) for stereo
        if data.ndim == 1:
            waveform = torch.from_numpy(data).unsqueeze(0)  # (1, n_samples)
        else:
            waveform = torch.from_numpy(data.T)  # (channels, n_samples)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # Pad or truncate to fixed length
        waveform = self._fix_length(waveform)

        return waveform

    def _fix_length(self, waveform: torch.Tensor) -> torch.Tensor:
        """Pad or truncate waveform to exactly n_samples.

        Args:
            waveform: Tensor of shape (1, T).

        Returns:
            Tensor of shape (1, n_samples).
        """
        length = waveform.shape[1]
        if length >= self.n_samples:
            # Truncate from the beginning
            waveform = waveform[:, :self.n_samples]
        else:
            # Zero-pad at the end
            pad_amount = self.n_samples - length
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        return waveform

    def compute_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute log-Mel spectrogram from waveform.

        Args:
            waveform: Tensor of shape (1, n_samples).

        Returns:
            Log-Mel spectrogram of shape (1, n_mels, T) where T depends on
            hop_length. With default settings: (1, 128, 251).
        """
        mel_spec = self.mel_transform(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)

        if self.mel_config.normalize:
            # Per-sample zero-mean unit-variance normalization
            mean = mel_spec_db.mean()
            std = mel_spec_db.std()
            if std > 0:
                mel_spec_db = (mel_spec_db - mean) / std

        return mel_spec_db

    def get_raw_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        """Return raw waveform for WavLM (squeeze channel dim).

        WavLM expects input of shape (batch, samples) without channel dim.

        Args:
            waveform: Tensor of shape (1, n_samples).

        Returns:
            Tensor of shape (n_samples,).
        """
        return waveform.squeeze(0)

    def get_rawnet2_input(self, waveform: torch.Tensor) -> torch.Tensor:
        """Return raw waveform for RawNet2.

        RawNet2 expects (1, n_samples) — keep channel dimension.

        Args:
            waveform: Tensor of shape (1, n_samples).

        Returns:
            Tensor of shape (1, n_samples).
        """
        return waveform

    def process(self, file_path: str) -> dict:
        """Full preprocessing pipeline: load audio and produce all representations.

        Args:
            file_path: Path to audio file.

        Returns:
            Dictionary with keys:
            - 'mel_spectrogram': (1, n_mels, T) log-Mel spectrogram
            - 'waveform_wavlm': (n_samples,) raw waveform for WavLM
            - 'waveform_rawnet2': (1, n_samples) raw waveform for RawNet2
            - 'sample_rate': int
        """
        waveform = self.load_audio(file_path)

        return {
            "mel_spectrogram": self.compute_mel_spectrogram(waveform),
            "waveform_wavlm": self.get_raw_waveform(waveform),
            "waveform_rawnet2": self.get_rawnet2_input(waveform),
            "sample_rate": self.sample_rate,
        }

    @staticmethod
    def from_config(config) -> "AudioPreprocessor":
        """Create preprocessor from a top-level Config object.

        Supports both the full Config (which has a .audio: AudioConfig field)
        and the lighter DataConfig (uses target_sr and segment_length).

        Args:
            config: Config object (src.utils.config.Config or similar).

        Returns:
            AudioPreprocessor instance.
        """
        if hasattr(config, "audio"):
            return AudioPreprocessor(config.audio)

        # Fallback: build AudioConfig from DataConfig fields
        audio_cfg = AudioConfig(
            sample_rate=getattr(config, "target_sr", 16000),
            n_samples=getattr(config, "segment_length", 64000),
        )
        return AudioPreprocessor(audio_cfg)


# ──────────────────────────────────────────────────────────────────────────────
# Manifest builder
# ──────────────────────────────────────────────────────────────────────────────

def run_preprocessing(
    raw_dir: str = "data/raw",
    output_dir: str = "data/processed",
    manifest_path: str = "data/metadata/master_manifest.csv",
    num_workers: int = 4,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> int:
    """Scan the raw data directory tree and write the master manifest CSV.

    Determines labels, domain, and source dataset from the directory structure:

        raw_dir/{domain}/{real_or_fake}/{source_dataset}/{...}/{file}

    Examples:
        data/raw/voice/real/asvspoof2019/LA_T_1138215.flac → label=0, domain=voice
        data/raw/voice/fake/wavefake/melgan/LJ001.wav       → label=1, domain=voice
        data/raw/music/real/musiccaps/abcdef.wav            → label=0, domain=music
        data/raw/non_human/real/esc50/1-100032-A-0.wav      → label=0, domain=non_human
        data/raw/non_human/fake/elevenlabs_sfx/rain.mp3     → label=1, domain=non_human

    Splits are assigned reproducibly using an MD5 hash of the file path so
    that re-running produces identical splits.

    Args:
        raw_dir: Root of the raw data tree.
        output_dir: (Unused — files are not copied; manifest references raw paths.)
        manifest_path: Path to write the CSV to.
        num_workers: (Unused for manifest-only mode.)
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation (rest → test).

    Returns:
        Total number of manifest rows written.
    """
    import csv
    import hashlib
    from pathlib import Path as _Path

    AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}

    raw = _Path(raw_dir)
    if not raw.exists():
        raise FileNotFoundError(f"raw_dir not found: {raw_dir}")

    manifest = _Path(manifest_path)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    def _assign_split(fp: str) -> str:
        h = int(hashlib.md5(fp.encode()).hexdigest(), 16) % 100
        if h < int(train_ratio * 100):
            return "train"
        elif h < int((train_ratio + val_ratio) * 100):
            return "val"
        return "test"

    rows = []
    for audio_file in sorted(raw.rglob("*")):
        if not audio_file.is_file():
            continue
        if audio_file.suffix.lower() not in AUDIO_EXTS:
            continue

        # Parse path components relative to raw_dir
        rel_parts = audio_file.relative_to(raw).parts
        # rel_parts[0] = domain, rel_parts[1] = real|fake, rel_parts[2] = source_dataset
        # rel_parts[3+] = optional generator subdirs

        domain = rel_parts[0] if len(rel_parts) > 0 else "voice"
        label_str = rel_parts[1].lower() if len(rel_parts) > 1 else "real"
        source = rel_parts[2] if len(rel_parts) > 2 else "unknown"
        generator = rel_parts[3] if len(rel_parts) > 3 else ("human" if label_str == "real" else source)

        label = 0.0 if label_str in ("real", "bonafide", "genuine") else 1.0

        file_path_rel = str(audio_file.relative_to(_Path(raw_dir).parent))  # relative to data_root
        split = _assign_split(file_path_rel)

        rows.append({
            "sample_id": f"{source}_{audio_file.stem}",
            "file_path": file_path_rel.replace("\\", "/"),
            "label": label,
            "domain": domain,
            "source_dataset": source,
            "generator": generator,
            "split": split,
        })

    with open(manifest, "w", newline="") as fh:
        fieldnames = ["sample_id", "file_path", "label", "domain", "source_dataset", "generator", "split"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    import logging as _logging
    _logging.getLogger(__name__).info(
        "Manifest written: %d rows → %s", len(rows), manifest_path
    )
    print(f"Manifest written: {len(rows)} rows → {manifest_path}")
    return len(rows)

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

from src.config import AudioConfig, MelConfig


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
        """Create preprocessor from top-level Config object.

        Args:
            config: Config object with .audio attribute.

        Returns:
            AudioPreprocessor instance.
        """
        return AudioPreprocessor(config.audio)

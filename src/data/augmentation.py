"""
Codec robustness augmentation pipeline.

Simulates real-world audio degradation by applying lossy codec compression
during training. This is one of our key research contributions — showing
that the detector remains robust to compressed/degraded audio.

Supported codecs:
- MP3 at 32/64/128 kbps
- AAC at 32/64 kbps
- Opus at 6/12/24 kbps
- Simulated VoIP: 8kHz downsample + G.711 mu-law

Requires ffmpeg to be installed on the system.
"""

import io
import os
import random
import shutil
import subprocess
import tempfile
from typing import Optional

import numpy as np
import torch
import torchaudio


def _check_ffmpeg() -> bool:
    """Check if ffmpeg is available on the system.

    Returns:
        True if ffmpeg is found, False otherwise.
    """
    return shutil.which("ffmpeg") is not None


class CodecAugmentor:
    """Applies random codec compression to audio waveforms.

    Args:
        mp3_bitrates: List of MP3 bitrate options in kbps.
        aac_bitrates: List of AAC bitrate options in kbps.
        opus_bitrates: List of Opus bitrate options in kbps.
        voip_sample_rate: Target sample rate for VoIP simulation.
        probability: Probability of applying augmentation to each sample.
        sample_rate: Input audio sample rate.
        enabled: Whether augmentation is active (disabled during eval).
    """

    def __init__(
        self,
        mp3_bitrates: list = None,
        aac_bitrates: list = None,
        opus_bitrates: list = None,
        voip_sample_rate: int = 8000,
        probability: float = 0.5,
        sample_rate: int = 16000,
        enabled: bool = True,
    ):
        self.mp3_bitrates = mp3_bitrates or [32, 64, 128]
        self.aac_bitrates = aac_bitrates or [32, 64]
        self.opus_bitrates = opus_bitrates or [6, 12, 24]
        self.voip_sample_rate = voip_sample_rate
        self.probability = probability
        self.sample_rate = sample_rate
        self.enabled = enabled
        self.has_ffmpeg = _check_ffmpeg()

        if enabled and not self.has_ffmpeg:
            print(
                "WARNING: ffmpeg not found. Codec augmentation will be skipped.\n"
                "Install ffmpeg: https://ffmpeg.org/download.html\n"
                "  Windows: winget install ffmpeg\n"
                "  Ubuntu: sudo apt install ffmpeg\n"
                "  macOS: brew install ffmpeg\n"
                "  Colab: !apt install ffmpeg"
            )

    def _waveform_to_wav_bytes(self, waveform: torch.Tensor) -> bytes:
        """Convert a waveform tensor to WAV bytes in memory.

        Args:
            waveform: Tensor of shape (1, N) or (N,).

        Returns:
            WAV file content as bytes.
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        buffer = io.BytesIO()
        torchaudio.save(buffer, waveform, self.sample_rate, format="wav")
        return buffer.getvalue()

    def _wav_bytes_to_waveform(self, wav_bytes: bytes) -> torch.Tensor:
        """Convert WAV bytes back to a waveform tensor.

        Args:
            wav_bytes: WAV file content as bytes.

        Returns:
            Tensor of shape (1, N).
        """
        buffer = io.BytesIO(wav_bytes)
        waveform, sr = torchaudio.load(buffer, format="wav")
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        return waveform

    def _apply_ffmpeg_codec(
        self, waveform: torch.Tensor, codec: str, bitrate: int
    ) -> torch.Tensor:
        """Apply codec compression via ffmpeg subprocess.

        Pipes audio through ffmpeg: WAV → codec → WAV, entirely in memory
        using stdin/stdout to avoid temp files.

        Args:
            waveform: Input tensor of shape (1, N).
            codec: Codec name ('libmp3lame', 'aac', 'libopus').
            bitrate: Target bitrate in kbps.

        Returns:
            Compressed-then-decoded waveform tensor of shape (1, N).
        """
        wav_bytes = self._waveform_to_wav_bytes(waveform)

        # Determine output format for intermediate encoding
        fmt_map = {
            "libmp3lame": "mp3",
            "aac": "adts",
            "libopus": "ogg",
        }
        out_fmt = fmt_map.get(codec, "mp3")

        # Encode: WAV → compressed format
        encode_cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "wav", "-i", "pipe:0",
            "-c:a", codec, "-b:a", f"{bitrate}k",
            "-f", out_fmt, "pipe:1",
        ]

        try:
            encode_proc = subprocess.run(
                encode_cmd,
                input=wav_bytes,
                capture_output=True,
                timeout=10,
            )
            if encode_proc.returncode != 0:
                return waveform  # Fall back to original on error

            compressed_bytes = encode_proc.stdout

            # Decode: compressed format → WAV
            decode_cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", out_fmt, "-i", "pipe:0",
                "-f", "wav", "-ar", str(self.sample_rate),
                "-ac", "1", "pipe:1",
            ]

            decode_proc = subprocess.run(
                decode_cmd,
                input=compressed_bytes,
                capture_output=True,
                timeout=10,
            )
            if decode_proc.returncode != 0:
                return waveform

            return self._wav_bytes_to_waveform(decode_proc.stdout)

        except (subprocess.TimeoutExpired, Exception):
            return waveform  # Fall back to original on any error

    def apply_mp3(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply MP3 compression at a random bitrate.

        Args:
            waveform: Input tensor of shape (1, N).

        Returns:
            MP3-compressed waveform.
        """
        bitrate = random.choice(self.mp3_bitrates)
        return self._apply_ffmpeg_codec(waveform, "libmp3lame", bitrate)

    def apply_aac(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply AAC compression at a random bitrate.

        Args:
            waveform: Input tensor of shape (1, N).

        Returns:
            AAC-compressed waveform.
        """
        bitrate = random.choice(self.aac_bitrates)
        return self._apply_ffmpeg_codec(waveform, "aac", bitrate)

    def apply_opus(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply Opus compression at a random bitrate.

        Args:
            waveform: Input tensor of shape (1, N).

        Returns:
            Opus-compressed waveform.
        """
        bitrate = random.choice(self.opus_bitrates)
        return self._apply_ffmpeg_codec(waveform, "libopus", bitrate)

    def apply_voip(self, waveform: torch.Tensor) -> torch.Tensor:
        """Simulate VoIP degradation: downsample to 8kHz + G.711 mu-law + upsample.

        This simulates the quality loss from phone/VoIP transmission without
        needing a separate G.711 codec — mu-law encoding/decoding is done in
        numpy for reliability.

        Args:
            waveform: Input tensor of shape (1, N).

        Returns:
            VoIP-degraded waveform at original sample rate.
        """
        # Downsample to 8kHz
        downsampler = torchaudio.transforms.Resample(
            self.sample_rate, self.voip_sample_rate
        )
        downsampled = downsampler(waveform)

        # Apply mu-law encoding/decoding (simulates G.711 quantization)
        mu = 255
        audio_np = downsampled.numpy().squeeze()
        # Normalize to [-1, 1]
        max_val = np.abs(audio_np).max()
        if max_val > 0:
            audio_np = audio_np / max_val
        # Mu-law compress
        compressed = np.sign(audio_np) * np.log1p(mu * np.abs(audio_np)) / np.log1p(mu)
        # Quantize to 8-bit (256 levels, matching G.711)
        quantized = np.round(compressed * 128) / 128
        # Mu-law expand
        expanded = np.sign(quantized) * (1 / mu) * ((1 + mu) ** np.abs(quantized) - 1)
        # Restore scale
        if max_val > 0:
            expanded = expanded * max_val

        # Convert back to tensor and upsample
        voip_waveform = torch.from_numpy(expanded).float().unsqueeze(0)
        upsampler = torchaudio.transforms.Resample(
            self.voip_sample_rate, self.sample_rate
        )
        return upsampler(voip_waveform)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply random codec augmentation with configured probability.

        During training, randomly selects one codec and applies it.
        During evaluation (enabled=False), returns the original waveform.

        Args:
            waveform: Input tensor of shape (1, N).

        Returns:
            Possibly augmented waveform of shape (1, N).
        """
        if not self.enabled:
            return waveform

        if random.random() > self.probability:
            return waveform

        # Build list of available augmentations
        augmentations = [self.apply_voip]  # VoIP always available (no ffmpeg needed)

        if self.has_ffmpeg:
            augmentations.extend([self.apply_mp3, self.apply_aac, self.apply_opus])

        aug_fn = random.choice(augmentations)
        augmented = aug_fn(waveform)

        # Ensure output length matches input
        target_len = waveform.shape[1]
        if augmented.shape[1] > target_len:
            augmented = augmented[:, :target_len]
        elif augmented.shape[1] < target_len:
            pad = target_len - augmented.shape[1]
            augmented = torch.nn.functional.pad(augmented, (0, pad))

        return augmented

    @staticmethod
    def from_config(config) -> "CodecAugmentor":
        """Create augmentor from top-level Config object.

        Args:
            config: Config object with .augmentation and .audio attributes.

        Returns:
            CodecAugmentor instance.
        """
        aug_cfg = config.augmentation
        return CodecAugmentor(
            mp3_bitrates=aug_cfg.codec.mp3_bitrates,
            aac_bitrates=aug_cfg.codec.aac_bitrates,
            opus_bitrates=aug_cfg.codec.opus_bitrates,
            voip_sample_rate=aug_cfg.codec.voip_sample_rate,
            probability=aug_cfg.probability,
            sample_rate=config.audio.sample_rate,
            enabled=aug_cfg.enabled,
        )

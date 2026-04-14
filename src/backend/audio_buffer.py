"""
Rolling audio buffer with sliding-window extraction.

The browser extension streams small audio chunks (~100-250ms) over
the WebSocket. We accumulate them into a ring buffer large enough to
hold one full model window (e.g., 4 seconds), then periodically
extract the most recent window for inference.

Design:
    - Fixed-size ring buffer holds the last `capacity_seconds` of audio
    - Incoming chunks can be any size; we append and overwrite oldest
    - `extract_latest_window()` returns the last `window_seconds` of audio
    - Resampling happens once at ingestion time (browser SR -> model SR)
"""

from __future__ import annotations

import numpy as np


class RingBuffer:
    """Simple fixed-size float32 ring buffer for mono audio."""

    def __init__(self, capacity_samples: int):
        self._buf = np.zeros(capacity_samples, dtype=np.float32)
        self._capacity = capacity_samples
        self._write_pos = 0
        self._total_written = 0

    def append(self, samples: np.ndarray):
        """Append samples to the buffer, overwriting oldest if full."""
        samples = np.ascontiguousarray(samples, dtype=np.float32)
        n = len(samples)

        if n >= self._capacity:
            # Just keep the last capacity samples
            self._buf[:] = samples[-self._capacity:]
            self._write_pos = 0
            self._total_written += n
            return

        end = self._write_pos + n
        if end <= self._capacity:
            self._buf[self._write_pos:end] = samples
        else:
            # wrap around
            first = self._capacity - self._write_pos
            self._buf[self._write_pos:] = samples[:first]
            self._buf[:n - first] = samples[first:]

        self._write_pos = end % self._capacity
        self._total_written += n

    def latest(self, n_samples: int) -> np.ndarray:
        """Return the most recent n_samples, oldest-first."""
        if n_samples > self._capacity:
            raise ValueError(f"Requested {n_samples} samples but capacity is {self._capacity}")

        if self._total_written < n_samples:
            # Not enough data yet; return what we have, zero-padded at the start
            have = min(self._total_written, self._capacity)
            out = np.zeros(n_samples, dtype=np.float32)
            if have > 0:
                # reconstruct the "have" newest samples
                recent = self._read_last_n(have)
                out[-have:] = recent
            return out

        return self._read_last_n(n_samples)

    def _read_last_n(self, n: int) -> np.ndarray:
        """Read the n most recent samples from the ring."""
        start = (self._write_pos - n) % self._capacity
        if start + n <= self._capacity:
            return self._buf[start:start + n].copy()
        # wrap
        first = self._capacity - start
        return np.concatenate([self._buf[start:], self._buf[:n - first]])

    @property
    def samples_written(self) -> int:
        return self._total_written

    def clear(self):
        self._buf.fill(0)
        self._write_pos = 0
        self._total_written = 0


class AudioWindowBuffer:
    """
    High-level buffer that:
        - Accepts chunks at the browser's sample rate
        - Resamples to the model's sample rate on ingestion
        - Maintains a ring buffer sized for one model window
        - Tracks how much new audio has accumulated since the last
          inference so we can decide when to run the next one (hop)
    """

    def __init__(
        self,
        source_sample_rate: int,
        target_sample_rate: int,
        window_seconds: float = 4.0,
        hop_seconds: float = 1.0,
    ):
        self.source_sr = source_sample_rate
        self.target_sr = target_sample_rate
        self.window_samples = int(window_seconds * target_sample_rate)
        self.hop_samples = int(hop_seconds * target_sample_rate)

        # Buffer holds slightly more than one window to smooth over boundaries
        capacity = int(window_seconds * 1.5 * target_sample_rate)
        self._ring = RingBuffer(capacity)

        # Samples accumulated (at target SR) since last window was extracted
        self._samples_since_last_extract = 0

        # Lightweight polyphase resampler state (optional); for simplicity we
        # do per-chunk resampling via numpy. If source==target, no resample.
        self._resample_ratio = target_sample_rate / source_sample_rate

    def append_chunk(self, samples: np.ndarray):
        """
        Append a chunk of float32 audio from the browser.
        Automatically resamples if source_sr != target_sr.
        """
        if samples.ndim > 1:
            # downmix to mono
            samples = samples.mean(axis=-1) if samples.shape[-1] <= 2 else samples.mean(axis=0)

        resampled = self._resample(samples) if self.source_sr != self.target_sr else samples

        self._ring.append(resampled)
        self._samples_since_last_extract += len(resampled)

    def _resample(self, samples: np.ndarray) -> np.ndarray:
        """Linear resampling (good enough for detection; faster than polyphase)."""
        if len(samples) == 0:
            return samples
        new_len = int(round(len(samples) * self._resample_ratio))
        if new_len <= 0:
            return np.array([], dtype=np.float32)
        # np.interp is linear; fine for our purposes
        old_x = np.linspace(0, 1, len(samples), dtype=np.float32)
        new_x = np.linspace(0, 1, new_len, dtype=np.float32)
        return np.interp(new_x, old_x, samples).astype(np.float32)

    def ready_for_inference(self) -> bool:
        """True if enough new audio has accumulated for the next inference hop."""
        has_full_window = self._ring.samples_written >= self.window_samples
        has_hop = self._samples_since_last_extract >= self.hop_samples
        return has_full_window and has_hop

    def extract_window(self) -> np.ndarray:
        """Get the most recent `window_samples` at target sample rate."""
        self._samples_since_last_extract = 0
        return self._ring.latest(self.window_samples)

    def clear(self):
        self._ring.clear()
        self._samples_since_last_extract = 0

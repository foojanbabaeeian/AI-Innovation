"""
Branch 3 -- Raw Waveform Branch (SincNet + Residual Blocks)

Operates directly on time-domain audio using learnable sinc bandpass
filters. Detects phase artifacts, codec compression patterns, and
fine-grained temporal anomalies that frequency-domain representations
may smooth over.

Architecture inspired by RawNet2 (Tak et al., 2021) and SincNet
(Ravanelli & Bengio, 2018).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SincConv(nn.Module):
    """
    Learnable sinc-function bandpass filters operating on raw waveforms.
    Each filter is parameterized by a center frequency and bandwidth,
    both learned during training.
    """

    def __init__(
        self,
        out_channels: int = 70,
        kernel_size: int = 251,
        sample_rate: int = 16000,
        min_low_hz: float = 50.0,
        min_band_hz: float = 50.0,
    ):
        super().__init__()
        assert kernel_size % 2 != 0, "Kernel size must be odd"

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # Initialize filter frequencies using mel scale
        low_hz = min_low_hz
        high_hz = sample_rate / 2 - (min_low_hz + min_band_hz)
        mel_low = 2595.0 * math.log10(1.0 + low_hz / 700.0)
        mel_high = 2595.0 * math.log10(1.0 + high_hz / 700.0)
        mel_points = torch.linspace(mel_low, mel_high, out_channels + 1)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)

        self.low_hz_ = nn.Parameter(hz_points[:-1].unsqueeze(1))
        self.band_hz_ = nn.Parameter((hz_points[1:] - hz_points[:-1]).unsqueeze(1))

        # Hamming window
        n = (kernel_size - 1) / 2.0
        self.register_buffer(
            "window",
            0.54 - 0.46 * torch.cos(2.0 * math.pi * torch.arange(-n, 0) / kernel_size),
        )
        self.register_buffer(
            "n_", (2 * math.pi * torch.arange(-n, 0).view(1, -1)) / sample_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz_),
            max=self.sample_rate / 2,
        )

        # Sinc filters
        f_low = torch.sin(high * self.n_) / (self.n_ / 2 + 1e-8) * self.window
        f_high = torch.sin(low * self.n_) / (self.n_ / 2 + 1e-8) * self.window
        band_pass_left = (f_low - f_high) / (2 * self.sample_rate)

        # Symmetric filter — center is already (out_channels, 1).
        band_pass_center = (high - low) / self.sample_rate
        band_pass = torch.cat(
            [band_pass_left, band_pass_center, band_pass_left.flip(dims=[1])],
            dim=1,
        )
        band_pass = band_pass / (band_pass.abs().sum(dim=1, keepdim=True) + 1e-8)

        filters = band_pass.unsqueeze(1)  # (out_channels, 1, kernel_size)
        return F.conv1d(x, filters, stride=1, padding=self.kernel_size // 2)


class ResBlock1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(0.3),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(0.3),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class RawNetBranch(nn.Module):
    """
    SincNet front-end + residual 1D-CNN backbone.

    Input:  (batch, 1, num_samples)
    Output: (batch, embed_dim) -- 128-d embedding
    """

    def __init__(self, sample_rate: int = 16000, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim

        self.sinc_conv = SincConv(
            out_channels=70,
            kernel_size=251,
            sample_rate=sample_rate,
        )
        self.sinc_pool = nn.MaxPool1d(kernel_size=3, stride=3)
        self.sinc_bn = nn.BatchNorm1d(70)

        self.block1 = nn.Sequential(
            nn.Conv1d(70, 128, kernel_size=3, padding=1, bias=False),
            ResBlock1D(128),
            ResBlock1D(128),
            nn.MaxPool1d(3),
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1, bias=False),
            ResBlock1D(256),
            ResBlock1D(256),
            nn.MaxPool1d(3),
        )

        self.block3 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1, bias=False),
            ResBlock1D(256),
            ResBlock1D(256),
            nn.AdaptiveAvgPool1d(1),
        )

        self.fc = nn.Linear(256, embed_dim)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: (batch, 1, num_samples)
        x = self.sinc_conv(waveform)  # (batch, 70, num_samples)
        x = self.sinc_pool(torch.abs(x))
        x = self.sinc_bn(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = x.squeeze(-1)  # (batch, 256)
        return self.fc(x)  # (batch, embed_dim)

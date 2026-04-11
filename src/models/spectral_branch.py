"""
Branch 1 -- Spectral CNN (ResNet-based)

Operates on log-mel spectrograms to detect formant artifacts,
phase inconsistencies, and vocoder residuals introduced by
neural audio synthesis models.
"""

import torch
import torch.nn as nn
import torchaudio.transforms as T


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class SpectralBranch(nn.Module):
    """
    Converts raw waveform -> log-mel spectrogram -> ResNet feature embedding.

    Input:  (batch, 1, num_samples)  -- raw waveform
    Output: (batch, embed_dim)       -- 128-d embedding vector
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        embed_dim: int = 128,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # On-the-fly mel spectrogram extraction
        self.mel_spec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80)

        # ResNet backbone: 1-channel input (single spectrogram)
        self.conv_stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(32, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, embed_dim)

    def _make_layer(
        self, in_channels: int, out_channels: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: (batch, 1, num_samples)
        x = waveform.squeeze(1)  # (batch, num_samples)
        x = self.mel_spec(x)  # (batch, n_mels, time_frames)
        x = self.amplitude_to_db(x)  # log scale
        x = x.unsqueeze(1)  # (batch, 1, n_mels, time_frames)

        x = self.conv_stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.flatten(1)  # (batch, 256)
        x = self.fc(x)  # (batch, embed_dim)
        return x

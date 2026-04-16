"""
Branch 2 -- Self-Supervised Learning Embeddings (wav2vec 2.0 / WavLM)

Uses a pretrained transformer encoder fine-tuned on the detection task.
Captures long-range prosodic irregularities and temporal coherence issues
that spectral snapshots miss.
"""

import torch
import torch.nn as nn
from transformers import WavLMModel, Wav2Vec2Model


class SSLBranch(nn.Module):
    """
    Extracts contextualized embeddings from a pretrained SSL audio model
    and projects them to a fixed-size embedding for fusion.

    Input:  (batch, 1, num_samples)  -- raw waveform at 16 kHz
    Output: (batch, embed_dim)       -- 128-d embedding vector
    """

    def __init__(
        self,
        model_name: str = "microsoft/wavlm-base-plus",
        embed_dim: int = 128,
        freeze_encoder: bool = False,
        freeze_feature_extractor: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Load pretrained SSL model
        if "wavlm" in model_name.lower():
            self.ssl_model = WavLMModel.from_pretrained(model_name)
        else:
            self.ssl_model = Wav2Vec2Model.from_pretrained(model_name)

        ssl_hidden_size = self.ssl_model.config.hidden_size  # typically 768

        # Freeze strategies for fine-tuning efficiency
        if freeze_encoder:
            for param in self.ssl_model.parameters():
                param.requires_grad = False
        elif freeze_feature_extractor:
            self.ssl_model.feature_extractor._freeze_parameters()

        # Weighted layer aggregation: learn which transformer layers matter most
        num_layers = self.ssl_model.config.num_hidden_layers + 1  # +1 for CNN feature output
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)

        # Projection head
        self.projector = nn.Sequential(
            nn.LayerNorm(ssl_hidden_size),
            nn.Linear(ssl_hidden_size, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: (batch, 1, num_samples)
        x = waveform.squeeze(1)  # (batch, num_samples) -- SSL models expect 1D input

        # WavLM's internal attention can overflow fp16; run in fp32
        # (it's frozen so this costs no extra gradient memory).
        with torch.amp.autocast(device_type="cuda", enabled=False):
            outputs = self.ssl_model(x.float(), output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple of (batch, time, 768)

        # Weighted sum across all transformer layers
        stacked = torch.stack(hidden_states, dim=0)  # (num_layers, batch, time, hidden)
        weights = torch.softmax(self.layer_weights, dim=0)
        weights = weights.view(-1, 1, 1, 1)
        weighted = (stacked * weights).sum(dim=0)  # (batch, time, hidden)

        # Mean pooling over time
        pooled = weighted.mean(dim=1)  # (batch, hidden)

        return self.projector(pooled)  # (batch, embed_dim)

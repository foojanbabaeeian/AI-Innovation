"""
Multi-Branch Attention Fusion Model

Fuses embeddings from all three branches (spectral, SSL, raw waveform)
using cross-branch multi-head attention, then produces:

1. Regression head  -- continuous AI probability per segment [0.0, 1.0]
   Captures *degree* of AI content (partially edited audio scores ~0.3-0.7)
2. Classification head -- categorical label (real / AI / mixed)
   Derived from segment-level regression scores aggregated over a clip

This dual-head design handles the real-world case where audio contains
both AI-generated and authentic portions (e.g., AI vocals over real
instrumentals, or AI-edited segments spliced into genuine recordings).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.spectral_branch import SpectralBranch
from src.models.ssl_branch import SSLBranch
from src.models.rawnet_branch import RawNetBranch


class CrossBranchAttention(nn.Module):
    """
    Multi-head attention across branch embeddings.
    Each branch embedding is treated as a token; attention learns
    which branches are most informative for each input.
    """

    def __init__(self, embed_dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, branch_embeddings: torch.Tensor) -> torch.Tensor:
        # branch_embeddings: (batch, num_branches, embed_dim)
        attn_out, _ = self.attn(branch_embeddings, branch_embeddings, branch_embeddings)
        x = self.norm(branch_embeddings + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x  # (batch, num_branches, embed_dim)


class MultiBranchFusionModel(nn.Module):
    """
    Full detection model: three branches -> attention fusion -> dual heads.

    Outputs:
        regression_score: (batch,)  -- P(AI) in [0, 1] per segment
        class_logits:     (batch, 3) -- logits for [real, mixed, ai]

    Training uses both heads jointly:
        L = alpha * MSE(regression_score, ai_ratio) + (1 - alpha) * CE(class_logits, class_label)

    Labels:
        ai_ratio:    float in [0, 1] -- fraction of AI content in segment
        class_label: 0 = real (ai_ratio < 0.2), 1 = mixed (0.2 <= ai_ratio <= 0.8), 2 = AI (ai_ratio > 0.8)
    """

    # Canonical branch order; forward() always emits embeddings in this order.
    BRANCH_ORDER = ("spectral", "ssl", "rawnet")

    def __init__(
        self,
        sample_rate: int = 16000,
        embed_dim: int = 128,
        num_attention_heads: int = 4,
        num_attention_layers: int = 2,
        num_classes: int = 3,
        ssl_model_name: str = "microsoft/wavlm-base-plus",
        freeze_ssl_feature_extractor: bool = True,
        freeze_ssl_encoder: bool = False,
        dropout: float = 0.1,
        disable_branches: list[str] | None = None,
        fusion_method: str = "attention",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # Validate ablation knobs
        disabled = set(disable_branches or [])
        unknown = disabled - set(self.BRANCH_ORDER)
        if unknown:
            raise ValueError(f"Unknown branches in disable_branches: {unknown}. "
                             f"Valid: {self.BRANCH_ORDER}")
        self.disabled_branches = disabled
        self.active_branches = tuple(b for b in self.BRANCH_ORDER if b not in disabled)
        if not self.active_branches:
            raise ValueError("At least one branch must be enabled.")

        if fusion_method not in ("attention", "concat", "average"):
            raise ValueError(f"fusion_method must be one of "
                             f"'attention', 'concat', 'average'; got {fusion_method!r}")
        self.fusion_method = fusion_method

        # Only instantiate branches that are active (saves params + compute on ablations).
        if "spectral" in self.active_branches:
            self.spectral_branch = SpectralBranch(
                sample_rate=sample_rate,
                embed_dim=embed_dim,
            )
        if "ssl" in self.active_branches:
            self.ssl_branch = SSLBranch(
                model_name=ssl_model_name,
                embed_dim=embed_dim,
                freeze_encoder=freeze_ssl_encoder,
                freeze_feature_extractor=freeze_ssl_feature_extractor,
            )
        if "rawnet" in self.active_branches:
            self.rawnet_branch = RawNetBranch(
                sample_rate=sample_rate,
                embed_dim=embed_dim,
            )

        # Cross-branch attention is only built when we actually use it AND when
        # there are >=2 active branches (attention over 1 token is a no-op).
        n_active = len(self.active_branches)
        if fusion_method == "attention" and n_active >= 2:
            self.attention_layers = nn.ModuleList([
                CrossBranchAttention(embed_dim, num_attention_heads, dropout)
                for _ in range(num_attention_layers)
            ])
        else:
            self.attention_layers = nn.ModuleList()

        # Head input dimension depends on fusion strategy.
        if fusion_method == "average" or n_active == 1:
            fused_dim = embed_dim
        else:  # attention or concat over multiple branches -> flattened
            fused_dim = embed_dim * n_active

        # Regression head: continuous AI probability [0, 1]
        self.regression_head = nn.Sequential(
            nn.Linear(fused_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),
        )

        # Classification head: real / mixed / AI
        self.classification_head = nn.Sequential(
            nn.Linear(fused_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, waveform: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            waveform: (batch, 1, num_samples) raw audio

        Returns:
            dict with:
                "regression_score": (batch,) -- P(AI) per segment
                "class_logits": (batch, num_classes) -- [real, mixed, ai]
                "branch_embeddings": (batch, n_active, embed_dim) -- for analysis
        """
        # Extract per-branch embeddings for ACTIVE branches only.
        embeddings = []
        if "spectral" in self.active_branches:
            embeddings.append(self.spectral_branch(waveform))
        if "ssl" in self.active_branches:
            embeddings.append(self.ssl_branch(waveform))
        if "rawnet" in self.active_branches:
            embeddings.append(self.rawnet_branch(waveform))

        # Stack as tokens: (batch, n_active, embed_dim)
        branch_tokens = torch.stack(embeddings, dim=1)

        # Fusion
        n_active = len(self.active_branches)
        if n_active == 1:
            # Single branch: bypass fusion entirely, use the one embedding.
            fused = branch_tokens.squeeze(1)
        elif self.fusion_method == "attention":
            for attn_layer in self.attention_layers:
                branch_tokens = attn_layer(branch_tokens)
            fused = branch_tokens.reshape(branch_tokens.size(0), -1)
        elif self.fusion_method == "concat":
            fused = branch_tokens.reshape(branch_tokens.size(0), -1)
        else:  # "average"
            fused = branch_tokens.mean(dim=1)

        regression_score = self.regression_head(fused).squeeze(-1)  # (batch,)
        class_logits = self.classification_head(fused)              # (batch, num_classes)

        return {
            "regression_score": regression_score,
            "class_logits": class_logits,
            "branch_embeddings": branch_tokens,
        }


class DualHeadLoss(nn.Module):
    """
    Combined loss for the dual-head architecture.

    L = alpha * MSE(pred_score, ai_ratio) + (1 - alpha) * CE(class_logits, class_label)

    Class labels are derived from ai_ratio:
        real:  ai_ratio < threshold_low  (default 0.2)
        mixed: threshold_low <= ai_ratio <= threshold_high
        AI:    ai_ratio > threshold_high (default 0.8)
    """

    def __init__(
        self,
        alpha: float = 0.5,
        threshold_low: float = 0.2,
        threshold_high: float = 0.8,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)

    def derive_class_labels(self, ai_ratio: torch.Tensor) -> torch.Tensor:
        """Convert continuous ai_ratio to discrete class labels."""
        labels = torch.ones_like(ai_ratio, dtype=torch.long)  # default: mixed (1)
        labels[ai_ratio < self.threshold_low] = 0   # real
        labels[ai_ratio > self.threshold_high] = 2  # AI
        return labels

    def forward(
        self,
        regression_score: torch.Tensor,
        class_logits: torch.Tensor,
        ai_ratio: torch.Tensor,
        class_label: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            regression_score: (batch,) model regression output
            class_logits: (batch, 3) model classification output
            ai_ratio: (batch,) ground truth AI content ratio [0, 1]
            class_label: (batch,) optional explicit class labels; if None, derived from ai_ratio
        """
        reg_loss = self.mse(regression_score, ai_ratio)

        if class_label is None:
            class_label = self.derive_class_labels(ai_ratio)
        cls_loss = self.ce(class_logits, class_label)

        total = self.alpha * reg_loss + (1 - self.alpha) * cls_loss

        return {
            "total_loss": total,
            "regression_loss": reg_loss,
            "classification_loss": cls_loss,
        }

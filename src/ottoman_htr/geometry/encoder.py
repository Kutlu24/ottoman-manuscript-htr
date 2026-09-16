from __future__ import annotations

import torch
from torch import nn


class GeometricEncoder(nn.Module):
    """Projects a hand-crafted geometric feature vector (see geometry.features) into the same
    embedding space as the visual encoder's output, so the two can be fused."""

    def __init__(self, in_dim: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)

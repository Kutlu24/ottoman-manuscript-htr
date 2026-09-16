from __future__ import annotations

import torch
from torch import nn


class FuzzyFusionLayer(nn.Module):
    """Fuses per-timestep character-class membership distributions from multiple evidence
    sources (vision, geometry, ...) into one, via a learned convex combination of per-source
    weights -- a differentiable stand-in for a fuzzy-inference aggregation rule. Combines
    distributions, not raw embeddings, so each source's confidence stays inspectable
    (see docs/architecture.md)."""

    def __init__(self, n_sources: int):
        super().__init__()
        self.source_logits = nn.Parameter(torch.zeros(n_sources))

    def forward(self, source_memberships: list[torch.Tensor]) -> torch.Tensor:
        weights = torch.softmax(self.source_logits, dim=0)
        stacked = torch.stack(source_memberships, dim=0)  # (n_sources, B, T, n_classes)
        fused = (weights.view(-1, 1, 1, 1) * stacked).sum(dim=0)
        return fused / fused.sum(dim=-1, keepdim=True).clamp_min(1e-8)

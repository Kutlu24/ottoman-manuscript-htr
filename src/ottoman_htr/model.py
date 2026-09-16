from __future__ import annotations

import torch
from torch import nn

from .config import ModelConfig
from .fuzzy.layer import FuzzyFusionLayer
from .fuzzy.membership import evidence_to_membership
from .geometry.encoder import GeometricEncoder
from .vision.encoder import ImageEncoder


class OttomanManuscriptHTR(nn.Module):
    """Vision + geometry -> fuzzy-fused per-timestep character membership -> CTC decoding.

    Keeps the visual and geometric evidence as separate class-membership distributions for as
    long as possible, and lets FuzzyFusionLayer combine them, rather than collapsing into one
    fused feature vector right after the encoders (see docs/architecture.md)."""

    def __init__(self, config: ModelConfig, vocab_size: int):
        super().__init__()
        self.vision_encoder = ImageEncoder(config.vision.in_channels, config.vision.embed_dim)
        self.geometric_encoder = GeometricEncoder(config.geometry.feature_dim, config.geometry.embed_dim)
        self.vision_head = nn.Linear(config.vision.embed_dim, vocab_size)
        self.geometry_head = nn.Linear(config.geometry.embed_dim, vocab_size)
        self.fusion = FuzzyFusionLayer(n_sources=2)

    def forward(self, images: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        visual_feats = self.vision_encoder(images)  # (B, T, Dv)
        geo_embed = self.geometric_encoder(geometry)  # (B, Dg)
        t = visual_feats.shape[1]
        geo_feats = geo_embed.unsqueeze(1).expand(-1, t, -1)  # broadcast across timesteps

        vision_membership = evidence_to_membership(self.vision_head(visual_feats))
        geometry_membership = evidence_to_membership(self.geometry_head(geo_feats))

        fused = self.fusion([vision_membership, geometry_membership])  # (B, T, vocab)
        return torch.log(fused.clamp_min(1e-8))  # log-probs, ready for CTC


def binarize_images(images: torch.Tensor) -> torch.Tensor:
    """Per-image adaptive threshold (mean - one std of pixel value), turning a continuous
    grayscale crop into a hard ink/background mask -- the 'binarization' step of the classical
    `scan -> binarization -> OCR` pipeline (Model A in the three-way ablation,
    docs/architecture.md). Images are dark-ink-on-light-paper in [0, 1], so ink is the low tail."""
    flat = images.flatten(1)
    threshold = (flat.mean(dim=1) - flat.std(dim=1)).clamp(0, 1).view(-1, 1, 1, 1)
    return (images < threshold).float()


class SimpleCTCModel(nn.Module):
    """Baseline for the three-way ablation: image -> CNN -> CTC, with no geometry and no fuzzy
    fusion -- isolates what the geometric+fuzzy machinery in OttomanManuscriptHTR adds on top of
    a plain neural recognizer sharing the same vision backbone (see docs/architecture.md).

    `binarize=True` makes this Model A (`scan -> binarization -> OCR`); `binarize=False` makes it
    Model B (`raw image -> CNN -> CTC`, continuous grayscale, still no geometry/fuzzy)."""

    def __init__(self, vision_config, vocab_size: int, binarize: bool = False):
        super().__init__()
        self.binarize = binarize
        self.vision_encoder = ImageEncoder(vision_config.in_channels, vision_config.embed_dim)
        self.ctc_head = nn.Linear(vision_config.embed_dim, vocab_size)

    def forward(self, images: torch.Tensor, geometry: torch.Tensor | None = None) -> torch.Tensor:
        if self.binarize:
            images = binarize_images(images)
        feats = self.vision_encoder(images)  # (B, T, D)
        return torch.log_softmax(self.ctc_head(feats), dim=-1)


def build_model(model_kind: str, config: ModelConfig, vocab_size: int) -> nn.Module:
    """Constructs the model for one arm of the three-way ablation. `model_kind` is one of
    'binary' (A), 'neural' (B), 'fuzzy' (C, OttomanManuscriptHTR) -- see docs/architecture.md."""
    if model_kind == "fuzzy":
        return OttomanManuscriptHTR(config, vocab_size)
    if model_kind == "neural":
        return SimpleCTCModel(config.vision, vocab_size, binarize=False)
    if model_kind == "binary":
        return SimpleCTCModel(config.vision, vocab_size, binarize=True)
    raise ValueError(f"unknown model_kind {model_kind!r}, expected one of: binary, neural, fuzzy")

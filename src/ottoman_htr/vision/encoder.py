from __future__ import annotations

import torch
from torch import nn


class ImageEncoder(nn.Module):
    """CRNN-style CNN backbone: collapses a grayscale line image's height to a single row while
    keeping width as the recognition timestep axis, producing a (B, T, embed_dim) sequence."""

    def __init__(self, in_channels: int = 1, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.width_stride = 4
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # height /2, width /2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # height /4, width /4
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # height /8, width unchanged
            nn.Conv2d(128, embed_dim, 3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.conv(images)  # (B, C, H', W')
        feats = feats.mean(dim=2)  # collapse residual height -> (B, C, W')
        return feats.transpose(1, 2)  # (B, T, embed_dim)

    def output_length(self, input_width: torch.Tensor) -> torch.Tensor:
        """Maps raw pixel widths to the encoder's output timestep count, for CTC input_lengths."""
        return torch.div(input_width, self.width_stride, rounding_mode="floor").clamp(min=1)

"""Minimal external segmentation model factory."""

from __future__ import annotations

import torch
from torch import nn


class DirectLogitsSegmenter(nn.Sequential):
    """Small network whose output already follows the direct-logits contract."""

    def __init__(self, in_channels: int, num_classes: int, width: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, num_classes, kernel_size=1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return super().forward(image)


def create_model(
    *,
    in_channels: int,
    num_classes: int,
    pretrained: bool,
    width: int = 12,
) -> nn.Module:
    """Factory boundary used by ``model.factory`` in YAML."""
    if pretrained:
        raise ValueError("the example model does not provide pretrained weights")
    if min(in_channels, num_classes, width) <= 0:
        raise ValueError("channel counts and width must be positive")
    return DirectLogitsSegmenter(in_channels, num_classes, width)

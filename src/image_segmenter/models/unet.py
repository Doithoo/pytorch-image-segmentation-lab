"""A compact, readable U-Net for the primary learning path."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Sequential):
    """Two convolution, normalization, and activation layers."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class UpBlock(nn.Module):
    """Upsample, concatenate the matching skip, and refine features."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.block = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        value = self.up(value)
        if value.shape[-2:] != skip.shape[-2:]:
            raise ValueError("U-Net skip shapes do not match; use dimensions divisible by 16")
        return self.block(torch.cat([value, skip], dim=1))


class SmallUNet(nn.Module):
    """Four-level U-Net that returns logits at the input resolution."""

    def __init__(self, in_channels: int = 3, num_classes: int = 2, base_channels: int = 16) -> None:
        super().__init__()
        if in_channels <= 0 or num_classes <= 0 or base_channels <= 0:
            raise ValueError("U-Net channel counts must be positive")
        widths = [base_channels * factor for factor in (1, 2, 4, 8)]
        self.encoders = nn.ModuleList(
            [
                ConvBlock(in_channels, widths[0]),
                ConvBlock(widths[0], widths[1]),
                ConvBlock(widths[1], widths[2]),
                ConvBlock(widths[2], widths[3]),
            ]
        )
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(widths[3], widths[3] * 2)
        self.decoders = nn.ModuleList(
            [
                UpBlock(widths[3] * 2, widths[3], widths[3]),
                UpBlock(widths[3], widths[2], widths[2]),
                UpBlock(widths[2], widths[1], widths[1]),
                UpBlock(widths[1], widths[0], widths[0]),
            ]
        )
        self.head = nn.Conv2d(widths[0], num_classes, 1)
        self.checkpoint_metadata = {"architecture": "unet_small", "pretrained_weights": None}

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError(f"expected image tensor [B,C,H,W], got shape {tuple(image.shape)}")
        if image.shape[-2] % 16 or image.shape[-1] % 16:
            raise ValueError("U-Net input height and width must be divisible by 16")
        skips: list[torch.Tensor] = []
        value = image
        for encoder in self.encoders:
            value = encoder(value)
            skips.append(value)
            value = self.pool(value)
        value = self.bottleneck(value)
        for decoder, skip in zip(self.decoders, reversed(skips), strict=True):
            value = decoder(value, skip)
        return self.head(value)

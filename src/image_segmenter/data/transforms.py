"""Transforms that keep segmentation images and masks geometrically aligned."""

from __future__ import annotations

import random

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

from ..config import DataConfig


class ImageTransform:
    """Deterministic image-only preprocessing for inference."""

    def __init__(self, image_size: int, mean: list[float], std: list[float]) -> None:
        self.image_size = image_size
        self.mean = mean
        self.std = std

    def __call__(self, image: Image.Image) -> torch.Tensor:
        resized = F.resize(
            image.convert("RGB"),
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        return F.normalize(F.to_tensor(resized), self.mean, self.std)


class PairedTransform:
    """Resize and optionally flip an image/mask pair with shared randomness."""

    def __init__(
        self,
        image_size: int,
        mean: list[float],
        std: list[float],
        horizontal_flip: float = 0.0,
    ) -> None:
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.horizontal_flip = horizontal_flip

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.horizontal_flip:
            image = F.hflip(image)
            mask = F.hflip(mask)
        size = [self.image_size, self.image_size]
        image = F.resize(image, size, interpolation=InterpolationMode.BILINEAR, antialias=True)
        mask = F.resize(mask, size, interpolation=InterpolationMode.NEAREST)
        image_tensor = F.normalize(F.to_tensor(image), self.mean, self.std)
        mask_tensor = F.pil_to_tensor(mask).squeeze(0).to(torch.int64)
        return image_tensor, mask_tensor


def build_transform(config: DataConfig, *, training: bool) -> PairedTransform:
    """Build deterministic evaluation or stochastic training preprocessing."""
    return PairedTransform(
        config.image_size,
        config.normalize_mean,
        config.normalize_std,
        horizontal_flip=config.horizontal_flip if training else 0.0,
    )

"""Transforms that keep segmentation images and masks geometrically aligned."""

from __future__ import annotations

import random

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

from ..config import DataConfig

SizeValue = int | list[int] | tuple[int, int]


def normalize_size(value: SizeValue, *, name: str) -> tuple[int, int]:
    """Normalize a square or [height, width] value and reject invalid dimensions."""
    if isinstance(value, int) and not isinstance(value, bool):
        result = (value, value)
    elif isinstance(value, list | tuple) and len(value) == 2:
        result = value[0], value[1]
    else:
        raise ValueError(f"{name} must be a positive integer or [height, width]")
    if any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in result):
        raise ValueError(f"{name} dimensions must be positive integers")
    return int(result[0]), int(result[1])


class ImageTransform:
    """Deterministic image-only preprocessing for inference."""

    def __init__(self, image_size: SizeValue, mean: list[float], std: list[float]) -> None:
        self.output_size = normalize_size(image_size, name="image_size")
        self.mean = mean
        self.std = std

    def __call__(self, image: Image.Image) -> torch.Tensor:
        resized = F.resize(
            image.convert("RGB"),
            list(self.output_size),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        return F.normalize(F.to_tensor(resized), self.mean, self.std)


class PairedTransform:
    """Apply shared resize, crop, and flip geometry to an image/mask pair."""

    def __init__(
        self,
        image_size: SizeValue,
        mean: list[float],
        std: list[float],
        horizontal_flip: float = 0.0,
        *,
        resize_mode: str = "stretch",
        crop_size: SizeValue | None = None,
        ignore_index: int = 255,
    ) -> None:
        self.output_size = normalize_size(image_size, name="image_size")
        self.mean = mean
        self.std = std
        if not isinstance(horizontal_flip, int | float) or isinstance(horizontal_flip, bool):
            raise ValueError("horizontal_flip must be between 0 and 1")
        if not 0 <= horizontal_flip <= 1:
            raise ValueError("horizontal_flip must be between 0 and 1")
        self.horizontal_flip = float(horizontal_flip)
        if resize_mode not in {"stretch", "letterbox"}:
            raise ValueError("resize_mode must be stretch or letterbox")
        self.resize_mode = resize_mode
        self.crop_size = normalize_size(crop_size, name="crop_size") if crop_size is not None else None
        if self.crop_size is not None and any(
            crop > output for crop, output in zip(self.crop_size, self.output_size, strict=True)
        ):
            raise ValueError("crop_size cannot exceed image_size")
        if not isinstance(ignore_index, int) or isinstance(ignore_index, bool):
            raise ValueError("ignore_index must be an integer")
        self.ignore_index = ignore_index

    def _resize(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        if self.resize_mode == "stretch":
            size = list(self.output_size)
            return (
                F.resize(image, size, interpolation=InterpolationMode.BILINEAR, antialias=True),
                F.resize(mask, size, interpolation=InterpolationMode.NEAREST),
            )

        target_height, target_width = self.output_size
        scale = min(target_width / image.width, target_height / image.height)
        resized_width = max(1, min(target_width, round(image.width * scale)))
        resized_height = max(1, min(target_height, round(image.height * scale)))
        resize_shape = [resized_height, resized_width]
        image = F.resize(image, resize_shape, interpolation=InterpolationMode.BILINEAR, antialias=True)
        mask = F.resize(mask, resize_shape, interpolation=InterpolationMode.NEAREST)
        horizontal_padding = target_width - resized_width
        vertical_padding = target_height - resized_height
        padding = [
            horizontal_padding // 2,
            vertical_padding // 2,
            horizontal_padding - horizontal_padding // 2,
            vertical_padding - vertical_padding // 2,
        ]
        return F.pad(image, padding, fill=0), F.pad(mask, padding, fill=self.ignore_index)

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = self._resize(image, mask)
        if self.crop_size is not None:
            crop_height, crop_width = self.crop_size
            top = random.randint(0, image.height - crop_height)
            left = random.randint(0, image.width - crop_width)
            image = F.crop(image, top, left, crop_height, crop_width)
            mask = F.crop(mask, top, left, crop_height, crop_width)
        if random.random() < self.horizontal_flip:
            image = F.hflip(image)
            mask = F.hflip(mask)
        image_tensor = F.normalize(F.to_tensor(image), self.mean, self.std)
        mask_tensor = F.pil_to_tensor(mask).squeeze(0).to(torch.int64)
        return image_tensor, mask_tensor


def build_transform(config: DataConfig, *, training: bool, ignore_index: int = 255) -> PairedTransform:
    """Build deterministic evaluation or stochastic training preprocessing."""
    return PairedTransform(
        config.image_size,
        config.normalize_mean,
        config.normalize_std,
        horizontal_flip=config.horizontal_flip if training else 0.0,
        resize_mode=config.resize_mode,
        crop_size=config.crop_size if training else None,
        ignore_index=ignore_index,
    )

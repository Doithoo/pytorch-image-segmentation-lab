"""Deterministic external Dataset factory with shared paired transforms."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from image_segmenter.data import LabelSchema


class SyntheticShapesDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    def __init__(
        self,
        split: str,
        length: int,
        source_size: int,
        transform: Any,
        label_schema: LabelSchema,
    ) -> None:
        if length <= 0 or source_size < 6:
            raise ValueError("length must be positive and source_size must be at least 6")
        if label_schema.num_classes != 3:
            raise ValueError("this example expects exactly three classes")
        self.split = split
        self.length = length
        self.source_size = source_size
        self.transform = transform

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        size = self.source_size
        image = np.zeros((size, size, 3), dtype=np.uint8)
        mask = np.zeros((size, size), dtype=np.uint8)
        left, right = size // 3, 2 * size // 3
        mask[:, left:right] = 1
        mask[size // 4 : 3 * size // 4, right:] = 2
        image[mask == 1] = (40 + index % 20, 180, 60)
        image[mask == 2] = (40, 80, 200)
        image_tensor, mask_tensor = self.transform(Image.fromarray(image), Image.fromarray(mask))
        return image_tensor, mask_tensor, f"{self.split}-{index:03d}"


def create_dataset(
    *,
    split: str,
    data_dir: Path,
    manifest_dir: Path,
    transform: Any,
    label_schema: LabelSchema,
    length: int = 8,
    source_size: int = 48,
) -> Dataset[tuple[torch.Tensor, torch.Tensor, str]]:
    """Factory boundary used by ``data.factory`` in YAML."""
    del data_dir, manifest_dir
    return SyntheticShapesDataset(split, length, source_size, transform, label_schema)

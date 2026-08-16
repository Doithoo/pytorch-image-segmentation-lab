from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from image_segmenter.data import LabelSchema


class TinyDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    def __init__(self, split: str, num_classes: int) -> None:
        self.split = split
        self.num_classes = num_classes

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        image = torch.full((3, 16, 16), float(index), dtype=torch.float32)
        mask = torch.full((16, 16), index % self.num_classes, dtype=torch.long)
        return image, mask, f"{self.split}-{index}"


def create_dataset(
    *,
    split: str,
    data_dir: Path,
    manifest_dir: Path,
    transform: Any,
    label_schema: LabelSchema,
    length: int = 2,
) -> Dataset[tuple[torch.Tensor, torch.Tensor, str]]:
    del data_dir, manifest_dir, transform, length
    return TinyDataset(split, label_schema.num_classes)


def returns_object(**_: Any) -> object:
    return object()


class BadItemDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        del index
        return torch.zeros(3, 8, 8), torch.zeros(8, 8)


def bad_item(**_: Any) -> Dataset[tuple[torch.Tensor, torch.Tensor]]:
    return BadItemDataset()

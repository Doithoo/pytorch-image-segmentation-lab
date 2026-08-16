"""Manifest-backed semantic segmentation Dataset."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import PairedTransform


def map_mask(raw: torch.Tensor, label_map: dict[int, int], ignore_values: list[int], ignore_index: int) -> torch.Tensor:
    """Map raw mask values to class indices and a single ignore value."""
    known = set(label_map) | set(ignore_values)
    found = {int(value) for value in raw.unique().tolist()}
    unknown = sorted(found - known)
    if unknown:
        raise ValueError(f"mask contains unmapped values: {unknown}")
    mapped = torch.full_like(raw, ignore_index, dtype=torch.int64)
    for raw_value, class_index in label_map.items():
        mapped[raw == raw_value] = class_index
    return mapped


class SegmentationDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    """Load paired images and masks listed in a portable CSV manifest."""

    def __init__(
        self,
        manifest_path: str | Path,
        transform: PairedTransform,
        label_map: dict[int, int],
        ignore_values: list[int],
        ignore_index: int,
        max_samples: int | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"manifest not found: {self.manifest_path}")
        with self.manifest_path.open(newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        required = {"sample_id", "image_path", "mask_path"}
        if self.rows and not required <= set(self.rows[0]):
            missing = sorted(required - set(self.rows[0]))
            raise ValueError(f"manifest is missing columns: {missing}")
        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError("max_samples must be positive")
            self.rows = self.rows[:max_samples]
        if not self.rows:
            raise ValueError(f"manifest contains no samples: {self.manifest_path}")
        self.transform = transform
        self.label_map = label_map
        self.ignore_values = ignore_values
        self.ignore_index = ignore_index

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (self.manifest_path.parent / path).resolve()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.rows[index]
        image_path = self._resolve(row["image_path"])
        mask_path = self._resolve(row["mask_path"])
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
        with Image.open(mask_path) as source_mask:
            mask = source_mask.convert("L")
        image_tensor, raw_mask = self.transform(image, mask)
        mapped_mask = map_mask(raw_mask, self.label_map, self.ignore_values, self.ignore_index)
        return image_tensor, mapped_mask, row["sample_id"]

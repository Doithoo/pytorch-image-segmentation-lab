"""Manifest-backed semantic segmentation Dataset."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .masks import MaskDecoder, RGBEntries
from .schema import ClassDefinition, LabelSchema
from .transforms import PairedTransform


def map_mask(raw: torch.Tensor, label_map: dict[int, int], ignore_values: list[int], ignore_index: int) -> torch.Tensor:
    """Map raw mask values to class indices and a single ignore value."""
    schema = _schema_from_mapping(label_map, ignore_index)
    decoder = MaskDecoder(schema, indexed_map=label_map, ignore_values=ignore_values)
    try:
        return decoder(raw)
    except ValueError as exc:
        prefix = "unknown indexed mask values:"
        if str(exc).startswith(prefix):
            raise ValueError(f"mask contains unmapped values:{str(exc).removeprefix(prefix)}") from exc
        raise


def _schema_from_mapping(label_map: dict[int, int], ignore_index: int) -> LabelSchema:
    class_ids = sorted(set(label_map.values()))
    if class_ids != list(range(len(class_ids))):
        raise ValueError(f"mapped class ids must be contiguous from 0; got {class_ids}")
    classes = tuple(
        ClassDefinition(
            class_id,
            ("background", "pet")[class_id] if len(class_ids) == 2 else f"class_{class_id}",
            (32, 32, 32) if class_id == 0 else ((46, 204, 113) if class_id == 1 else _class_color(class_id)),
        )
        for class_id in class_ids
    )
    return LabelSchema(classes=classes, ignore_index=ignore_index)


def _class_color(class_id: int) -> tuple[int, int, int]:
    return ((class_id * 67) % 256, (class_id * 131) % 256, (class_id * 197) % 256)


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
        *,
        label_schema: LabelSchema | None = None,
        rgb_map: RGBEntries | None = None,
        ignore_colors: tuple[tuple[int, int, int], ...] = (),
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
        self.label_schema = label_schema or _schema_from_mapping(label_map, ignore_index)
        self.mask_decoder = MaskDecoder(
            self.label_schema,
            indexed_map=label_map,
            ignore_values=ignore_values,
            rgb_map=rgb_map,
            ignore_colors=ignore_colors,
        )

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
            decoded_mask = self.mask_decoder(source_mask)
        mask = Image.fromarray(np.asarray(decoded_mask, dtype=np.int32))
        image_tensor, mapped_mask = self.transform(image, mask)
        return image_tensor, mapped_mask, row["sample_id"]

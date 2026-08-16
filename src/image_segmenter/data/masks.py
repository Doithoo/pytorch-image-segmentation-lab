"""Decode common semantic-segmentation mask encodings without information loss."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import torch
from PIL import Image

from .schema import LabelSchema

RGB = tuple[int, int, int]
RGBEntries = Mapping[RGB, int] | Iterable[tuple[RGB, int]]


class MaskDecoder:
    """Map indexed or RGB source masks into one canonical LabelSchema."""

    def __init__(
        self,
        schema: LabelSchema,
        *,
        indexed_map: Mapping[int, int] | None = None,
        ignore_values: Iterable[int] = (),
        rgb_map: RGBEntries | None = None,
        ignore_colors: Iterable[RGB] = (),
    ) -> None:
        self.schema = schema
        self.indexed_map = dict(indexed_map or {item.id: item.id for item in schema.classes})
        self.ignore_values = frozenset(int(value) for value in ignore_values)
        if set(self.indexed_map) & self.ignore_values:
            raise ValueError("indexed mapping and ignore values must not overlap")
        self._validate_targets(self.indexed_map.values())

        entries = list(rgb_map.items() if isinstance(rgb_map, Mapping) else (rgb_map or ()))
        self.rgb_map: dict[RGB, int] = {}
        for color, class_id in entries:
            normalized = self._validate_color(color)
            previous = self.rgb_map.get(normalized)
            if previous is not None and previous != class_id:
                raise ValueError(f"RGB color mapping collision for {normalized}")
            self.rgb_map[normalized] = class_id
        self._validate_targets(self.rgb_map.values())
        self.ignore_colors = frozenset(self._validate_color(color) for color in ignore_colors)
        if set(self.rgb_map) & self.ignore_colors:
            raise ValueError("RGB mapping and ignore colors must not overlap")

    def _validate_targets(self, targets: Iterable[int]) -> None:
        valid = set(range(self.schema.num_classes))
        for class_id in targets:
            if not isinstance(class_id, int) or isinstance(class_id, bool) or class_id not in valid:
                raise ValueError(f"mask mapping targets unknown class id {class_id!r}")

    @staticmethod
    def _validate_color(color: object) -> RGB:
        if (
            not isinstance(color, tuple | list)
            or len(color) != 3
            or any(not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 255 for value in color)
        ):
            raise ValueError(f"invalid RGB mask color: {color!r}")
        return int(color[0]), int(color[1]), int(color[2])

    def __call__(self, mask: Image.Image | np.ndarray | torch.Tensor) -> torch.Tensor:
        """Decode a PIL image, NumPy array, or integer tensor to torch.long."""
        array = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
        if array.ndim == 2:
            return self._decode_indexed(array)
        if array.ndim == 3 and array.shape[2] == 3:
            return self._decode_rgb(array)
        raise ValueError(f"mask must be indexed [H,W] or RGB [H,W,3], got shape {array.shape}")

    def _decode_indexed(self, array: np.ndarray) -> torch.Tensor:
        if not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"indexed mask must contain integers, got dtype {array.dtype}")
        found = {int(value) for value in np.unique(array)}
        unknown = sorted(found - set(self.indexed_map) - set(self.ignore_values))
        if unknown:
            raise ValueError(f"unknown indexed mask values: {unknown}")
        mapped = np.full(array.shape, self.schema.ignore_index, dtype=np.int64)
        for raw_value, class_id in self.indexed_map.items():
            mapped[array == raw_value] = class_id
        return torch.from_numpy(mapped)

    def _decode_rgb(self, array: np.ndarray) -> torch.Tensor:
        if array.dtype != np.uint8:
            raise ValueError(f"RGB mask must use 8-bit channels, got dtype {array.dtype}")
        found = {tuple(int(value) for value in row) for row in np.unique(array.reshape(-1, 3), axis=0)}
        unknown = sorted(found - set(self.rgb_map) - set(self.ignore_colors))
        if unknown:
            raise ValueError(f"unknown RGB mask colors: {unknown}")
        mapped = np.full(array.shape[:2], self.schema.ignore_index, dtype=np.int64)
        for color, class_id in self.rgb_map.items():
            mapped[np.all(array == color, axis=-1)] = class_id
        return torch.from_numpy(mapped)

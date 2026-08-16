"""Validate paired image/mask directories and write portable manifests."""

from __future__ import annotations

import csv
import hashlib
import math
import os
import random
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MANIFEST_FIELDS = ("sample_id", "image_path", "mask_path", "split", "width", "height")
INVALID_FIELDS = ("sample_id", "path", "reason")


@dataclass(frozen=True)
class ManifestRow:
    """One validated segmentation sample."""

    sample_id: str
    image_path: str
    mask_path: str
    split: str
    width: int
    height: int


class ManifestError(ValueError):
    """Raised after all invalid dataset samples have been collected."""


def _indexed_files(root: Path, suffixes: set[str]) -> tuple[dict[str, Path], list[dict[str, str]]]:
    indexed: dict[str, Path] = {}
    invalid: list[dict[str, str]] = []
    if not root.is_dir():
        return indexed, [{"sample_id": "", "path": str(root), "reason": "directory not found"}]
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.suffix.lower() in suffixes):
        sample_id = path.relative_to(root).with_suffix("").as_posix()
        if sample_id in indexed:
            invalid.append({"sample_id": sample_id, "path": str(path), "reason": "duplicate relative path stem"})
        else:
            indexed[sample_id] = path
    return indexed, invalid


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_dict_rows(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _relative_path(path: Path, manifest_dir: Path) -> str:
    return Path(os.path.relpath(path, manifest_dir)).as_posix()


def prepare_paired_manifests(
    data_dir: str | Path,
    manifest_dir: str | Path,
    split_ratios: list[float],
    seed: int,
    label_map: dict[int, int],
    ignore_values: set[int],
) -> dict[str, Path]:
    """Validate paired directories and create deterministic train/valid/test CSVs."""
    source = Path(data_dir).resolve()
    output = Path(manifest_dir).resolve()
    if len(split_ratios) != 3 or any(value < 0 for value in split_ratios) or abs(sum(split_ratios) - 1.0) > 1e-9:
        raise ValueError("split_ratios must contain three non-negative values summing to 1")
    if set(label_map.values()) != {0, 1}:
        raise ValueError("label_map must map raw values to background 0 and foreground 1")
    if set(label_map) & ignore_values:
        raise ValueError("label_map and ignore_values must not overlap")

    images, invalid = _indexed_files(source / "images", IMAGE_SUFFIXES)
    masks, mask_index_errors = _indexed_files(source / "masks", {".png"})
    invalid.extend(mask_index_errors)
    all_sample_ids = sorted(set(images) | set(masks))
    valid_samples: list[tuple[str, Path, Path, int, int]] = []
    seen_content: dict[tuple[str, str], str] = {}
    foreground_values = {raw_value for raw_value, class_index in label_map.items() if class_index == 1}
    allowed_values = set(label_map) | ignore_values
    empty_foreground_count = 0

    for sample_id in all_sample_ids:
        image_path = images.get(sample_id)
        mask_path = masks.get(sample_id)
        if image_path is None:
            invalid.append({"sample_id": sample_id, "path": str(mask_path), "reason": "missing image"})
            continue
        if mask_path is None:
            invalid.append({"sample_id": sample_id, "path": str(image_path), "reason": "missing mask"})
            continue
        try:
            with Image.open(image_path) as image:
                image.load()
                width, height = image.size
            with Image.open(mask_path) as mask:
                mask.load()
                if mask.size != (width, height):
                    raise ValueError(f"size mismatch: image={(width, height)}, mask={mask.size}")
                mask_array = np.asarray(mask)
            if mask_array.ndim != 2:
                raise ValueError(f"mask must be single-channel, got shape {mask_array.shape}")
            found_values = {int(value) for value in np.unique(mask_array)}
            unexpected = sorted(found_values - allowed_values)
            if unexpected:
                raise ValueError(f"unexpected mask values: {unexpected}")
            if not found_values & set(label_map):
                raise ValueError("mask has no trainable pixels")
            if not found_values & foreground_values:
                empty_foreground_count += 1
            content_key = (_sha256(image_path), _sha256(mask_path))
            duplicate_of = seen_content.get(content_key)
            if duplicate_of is not None:
                raise ValueError(f"exact duplicate of {duplicate_of}")
            seen_content[content_key] = sample_id
            valid_samples.append((sample_id, image_path, mask_path, width, height))
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            invalid.append({"sample_id": sample_id, "path": str(mask_path), "reason": str(exc)})

    if invalid:
        _write_dict_rows(output / "invalid.csv", INVALID_FIELDS, invalid)
        noun = "sample" if len(invalid) == 1 else "samples"
        raise ManifestError(f"found {len(invalid)} invalid {noun}; see {output / 'invalid.csv'}")
    if not valid_samples:
        raise ManifestError("found no valid paired samples")
    if empty_foreground_count:
        noun = "sample has" if empty_foreground_count == 1 else "samples have"
        warnings.warn(f"{empty_foreground_count} {noun} no foreground; retained as negative data", UserWarning, stacklevel=2)

    random.Random(seed).shuffle(valid_samples)
    total = len(valid_samples)
    train_count = math.floor(total * split_ratios[0])
    valid_count = math.floor(total * split_ratios[1])
    boundaries = (train_count, train_count + valid_count, total)
    split_items = {
        "train": valid_samples[: boundaries[0]],
        "valid": valid_samples[boundaries[0] : boundaries[1]],
        "test": valid_samples[boundaries[1] : boundaries[2]],
    }

    result: dict[str, Path] = {}
    for split, items in split_items.items():
        rows = [
            asdict(
                ManifestRow(
                    sample_id=sample_id,
                    image_path=_relative_path(image_path, output),
                    mask_path=_relative_path(mask_path, output),
                    split=split,
                    width=width,
                    height=height,
                )
            )
            for sample_id, image_path, mask_path, width, height in items
        ]
        path = output / f"{split}.csv"
        _write_dict_rows(path, MANIFEST_FIELDS, rows)
        result[split] = path
    return result

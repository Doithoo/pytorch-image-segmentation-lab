"""Validate paired segmentation data and write portable split manifests."""

from __future__ import annotations

import csv
import hashlib
import math
import os
import random
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, UnidentifiedImageError

from .metadata import PreparedDatasetMetadata, write_dataset_metadata
from .schema import ClassDefinition, LabelSchema

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MANIFEST_FIELDS = (
    "sample_id",
    "image_path",
    "mask_path",
    "split",
    "width",
    "height",
    "breed",
    "source_split",
    "group_id",
)
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
    breed: str = ""
    source_split: str = ""
    group_id: str = ""


@dataclass(frozen=True)
class _ValidatedSample:
    sample_id: str
    image_path: Path
    mask_path: Path
    width: int
    height: int


class ManifestError(ValueError):
    """Raised after all invalid dataset samples have been collected."""


def _audit_exact_duplicates(rows: Sequence[Mapping[str, object]], root: Path) -> None:
    """Reject identical image/mask content appearing in different splits."""
    seen: dict[tuple[str, str], str] = {}
    for row in rows:
        image_path = Path(str(row["image_path"]))
        mask_path = Path(str(row["mask_path"]))
        if not image_path.is_absolute():
            image_path = (root / image_path).resolve()
        if not mask_path.is_absolute():
            mask_path = (root / mask_path).resolve()
        key = (_sha256(image_path), _sha256(mask_path))
        sample_id = str(row["sample_id"])
        duplicate_of = seen.get(key)
        if duplicate_of is not None and duplicate_of != sample_id:
            raise ManifestError(f"exact duplicate of {duplicate_of}: {sample_id}")
        if duplicate_of == sample_id:
            continue
        seen[key] = sample_id


def _indexed_files(root: Path, suffixes: set[str]) -> tuple[dict[str, Path], list[dict[str, str]]]:
    indexed: dict[str, Path] = {}
    invalid: list[dict[str, str]] = []
    if not root.is_dir():
        return indexed, [{"sample_id": "", "path": str(root), "reason": "directory not found"}]
    paths = sorted(item for item in root.rglob("*") if item.is_file() and item.suffix.lower() in suffixes)
    for path in paths:
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


def _write_dict_rows(path: Path, fieldnames: tuple[str, ...], rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _relative_path(path: Path, manifest_dir: Path) -> str:
    return Path(os.path.relpath(path, manifest_dir)).as_posix()


def _validate_source(
    source: Path,
    output: Path,
    label_map: dict[int, int],
    ignore_values: set[int],
) -> list[_ValidatedSample]:
    if set(label_map.values()) != {0, 1}:
        raise ValueError("label_map must map raw values to background 0 and foreground 1")
    if set(label_map) & ignore_values:
        raise ValueError("label_map and ignore_values must not overlap")
    images, invalid = _indexed_files(source / "images", IMAGE_SUFFIXES)
    masks, mask_errors = _indexed_files(source / "masks", {".png"})
    invalid.extend(mask_errors)
    valid_samples: list[_ValidatedSample] = []
    seen_content: dict[tuple[str, str], str] = {}
    foreground_values = {raw_value for raw_value, class_index in label_map.items() if class_index == 1}
    allowed_values = set(label_map) | ignore_values
    empty_foreground_count = 0

    for sample_id in sorted(set(images) | set(masks)):
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
            valid_samples.append(_ValidatedSample(sample_id, image_path, mask_path, width, height))
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
        warnings.warn(
            f"{empty_foreground_count} {noun} no foreground; retained as negative data", UserWarning, stacklevel=3
        )
    return valid_samples


def _write_metadata(
    output: Path,
    source: Path,
    dataset_type: str,
    seed: int | None,
    label_map: dict[int, int],
    ignore_values: set[int],
    paths: dict[str, Path],
    counts: dict[str, int],
    audit: dict[str, object] | None = None,
    *,
    provider_name: str | None = None,
    split_method_name: str | None = None,
    label_schema: LabelSchema | None = None,
) -> None:
    checksums = {split: _sha256(path) for split, path in paths.items()}
    metadata: dict[str, Any] = {
        "dataset": dataset_type,
        "data_dir": _relative_path(source, output),
        "seed": seed,
        "label_map": label_map,
        "ignore_values": sorted(ignore_values),
        "split_counts": counts,
        "manifest_sha256": checksums,
        "exact_duplicate_audit": "passed",
    }
    if dataset_type == "oxford-iiit-pet":
        metadata["source_url"] = "https://www.robots.ox.ac.uk/~vgg/data/pets/"
        metadata["license"] = "Creative Commons Attribution-ShareAlike 4.0 International"
    metadata.update(audit or {})
    (output / "source.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False))
    summary = [
        f"dataset: {dataset_type}",
        f"seed: {seed}",
        *(f"{split}: {counts[split]}" for split in ("train", "valid", "test")),
        f"label_map: {label_map}",
        f"ignore_values: {sorted(ignore_values)}",
        *(f"{split}_sha256: {checksums[split]}" for split in ("train", "valid", "test")),
        "exact_duplicate_audit: passed",
        *(f"{key}: {value}" for key, value in (audit or {}).items()),
    ]
    (output / "summary.txt").write_text("\n".join(summary) + "\n")
    provider = provider_name or ("oxford_pet" if dataset_type == "oxford-iiit-pet" else "paired_masks")
    split_method = split_method_name or (
        "official_test_stratified_train_valid" if provider == "oxford_pet" else "random"
    )
    if label_schema is None:
        label_schema = LabelSchema(
            classes=(
                ClassDefinition(0, "background", (32, 32, 32)),
                ClassDefinition(1, "pet", (46, 204, 113)),
            ),
            ignore_index=255,
        )
    prepared = PreparedDatasetMetadata.create(
        provider=provider,
        split_method=split_method,
        seed=seed,
        data_dir=_relative_path(source, output),
        manifests={split: _relative_path(path, output) for split, path in paths.items()},
        split_counts=counts,
        label_schema=label_schema,
        root=output,
        label_map=label_map,
        ignore_values=tuple(sorted(ignore_values)),
        audit={"exact_duplicate_audit": "passed", **(audit or {})},
    )
    write_dataset_metadata(output / "dataset.yaml", prepared)


def _write_manifests(
    output: Path,
    source: Path,
    split_items: dict[str, list[tuple[_ValidatedSample, str, str]]],
    dataset_type: str,
    seed: int,
    label_map: dict[int, int],
    ignore_values: set[int],
    audit: dict[str, object] | None = None,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    all_rows: list[dict[str, object]] = []
    for split in ("train", "valid", "test"):
        rows = [
            asdict(
                ManifestRow(
                    sample_id=sample.sample_id,
                    image_path=_relative_path(sample.image_path, output),
                    mask_path=_relative_path(sample.mask_path, output),
                    split=split,
                    width=sample.width,
                    height=sample.height,
                    breed=breed,
                    source_split=source_split,
                )
            )
            for sample, breed, source_split in split_items[split]
        ]
        all_rows.extend(rows)
        path = output / f"{split}.csv"
        _write_dict_rows(path, MANIFEST_FIELDS, rows)
        paths[split] = path
    _audit_exact_duplicates(all_rows, output)
    counts = {split: len(items) for split, items in split_items.items()}
    _write_metadata(output, source, dataset_type, seed, label_map, ignore_values, paths, counts, audit)
    return paths


def prepare_paired_manifests(
    data_dir: str | Path,
    manifest_dir: str | Path,
    split_ratios: list[float],
    seed: int,
    label_map: dict[int, int],
    ignore_values: set[int],
) -> dict[str, Path]:
    """Create deterministic random splits for a generic paired dataset."""
    source = Path(data_dir).resolve()
    output = Path(manifest_dir).resolve()
    if len(split_ratios) != 3 or any(value < 0 for value in split_ratios) or abs(sum(split_ratios) - 1.0) > 1e-9:
        raise ValueError("split_ratios must contain three non-negative values summing to 1")
    samples = _validate_source(source, output, label_map, ignore_values)
    random.Random(seed).shuffle(samples)
    train_count = math.floor(len(samples) * split_ratios[0])
    valid_count = math.floor(len(samples) * split_ratios[1])
    first = train_count
    second = train_count + valid_count
    split_samples = {
        "train": samples[:first],
        "valid": samples[first:second],
        "test": samples[second:],
    }
    empty_splits = [split for split, split_values in split_samples.items() if not split_values]
    if empty_splits:
        names = ", ".join(empty_splits)
        raise ManifestError(f"split ratios produce an empty split ({names}); add samples or adjust data.split_ratios")
    annotated = {
        split: [(sample, "", "provided") for sample in split_values] for split, split_values in split_samples.items()
    }
    return _write_manifests(output, source, annotated, "paired-directory", seed, label_map, ignore_values)


def _read_oxford_split(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Oxford split file not found: {path}")
    sample_ids: list[str] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) != 4:
            raise ManifestError(f"invalid Oxford split line {path}:{line_number}")
        sample_ids.append(parts[0])
    if len(sample_ids) != len(set(sample_ids)):
        raise ManifestError(f"duplicate sample IDs in Oxford split: {path}")
    return sample_ids


def _breed_name(sample_id: str) -> str:
    name, separator, suffix = sample_id.rpartition("_")
    if not separator or not suffix.isdigit() or not name:
        raise ManifestError(f"cannot derive Oxford breed from sample ID: {sample_id}")
    return name


def prepare_oxford_manifests(
    data_dir: str | Path,
    manifest_dir: str | Path,
    seed: int,
    label_map: dict[int, int],
    ignore_values: set[int],
) -> dict[str, Path]:
    """Preserve official test data and split trainval within every breed."""
    source = Path(data_dir).resolve()
    output = Path(manifest_dir).resolve()
    samples = _validate_source(source, output, label_map, ignore_values)
    by_id = {sample.sample_id: sample for sample in samples}
    trainval_ids = _read_oxford_split(source / "annotations" / "trainval.txt")
    test_ids = _read_oxford_split(source / "annotations" / "test.txt")
    if set(trainval_ids) & set(test_ids):
        raise ManifestError("Oxford trainval and test splits overlap")
    listed = set(trainval_ids) | set(test_ids)
    missing_files = sorted(listed - set(by_id))
    if missing_files:
        raise ManifestError(f"Oxford split membership mismatch; missing_files={missing_files[:5]}")
    unlisted = sorted(set(by_id) - listed)

    groups: dict[str, list[str]] = {}
    for sample_id in trainval_ids:
        groups.setdefault(_breed_name(sample_id), []).append(sample_id)
    train_items: list[tuple[_ValidatedSample, str, str]] = []
    valid_items: list[tuple[_ValidatedSample, str, str]] = []
    for breed in sorted(groups):
        breed_ids = sorted(groups[breed])
        random.Random(f"{seed}:{breed}").shuffle(breed_ids)
        train_count = 1 if len(breed_ids) == 1 else min(len(breed_ids) - 1, max(1, math.floor(len(breed_ids) * 0.8)))
        train_items.extend((by_id[sample_id], breed, "trainval") for sample_id in breed_ids[:train_count])
        valid_items.extend((by_id[sample_id], breed, "trainval") for sample_id in breed_ids[train_count:])
    test_items = [(by_id[sample_id], _breed_name(sample_id), "test") for sample_id in test_ids]
    split_items = {
        "train": sorted(train_items, key=lambda item: item[0].sample_id),
        "valid": sorted(valid_items, key=lambda item: item[0].sample_id),
        "test": test_items,
    }
    audit: dict[str, object] = {
        "unlisted_pair_count": len(unlisted),
        "unlisted_pair_examples": unlisted[:10],
    }
    return _write_manifests(
        output,
        source,
        split_items,
        "oxford-iiit-pet",
        seed,
        label_map,
        ignore_values,
        audit,
    )

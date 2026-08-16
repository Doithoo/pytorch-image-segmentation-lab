"""Built-in preparation providers for common segmentation directory layouts."""

from __future__ import annotations

import csv
import random
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from .manifest import (
    MANIFEST_FIELDS,
    ManifestError,
    ManifestRow,
    _relative_path,
    _validate_source,
    _write_dict_rows,
    _write_manifests,
    _write_metadata,
    prepare_oxford_manifests,
    prepare_paired_manifests,
)
from .schema import LabelSchema


def _reject_unknown_params(params: dict[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise ValueError(f"unknown dataset provider parameter: {unknown[0]}")


def prepare_oxford_provider(**kwargs: Any) -> dict[str, Path]:
    params = dict(kwargs.pop("params", {}))
    _reject_unknown_params(params, set())
    kwargs.pop("split_ratios")
    return prepare_oxford_manifests(**kwargs)


def prepare_paired_provider(**kwargs: Any) -> dict[str, Path]:
    params = dict(kwargs.pop("params", {}))
    _reject_unknown_params(params, {"group_regex"})
    group_regex = params.get("group_regex")
    if group_regex is None:
        return prepare_paired_manifests(**kwargs)

    source = Path(kwargs["data_dir"]).resolve()
    output = Path(kwargs["manifest_dir"]).resolve()
    ratios = kwargs["split_ratios"]
    seed = kwargs["seed"]
    label_map = kwargs["label_map"]
    ignore_values = kwargs["ignore_values"]
    samples = _validate_source(source, output, label_map, ignore_values)
    pattern = re.compile(str(group_regex))
    groups: dict[str, list[Any]] = {}
    for sample in samples:
        match = pattern.search(sample.sample_id)
        if match is None:
            raise ManifestError(f"group_regex does not match sample_id: {sample.sample_id}")
        group_id = match.group(1) if match.lastindex else match.group(0)
        groups.setdefault(group_id, []).append(sample)
    if len(groups) < 3:
        raise ManifestError("group-preserving split requires at least three groups")
    shuffled = sorted(groups)
    random.Random(seed).shuffle(shuffled)
    target = [len(samples) * ratio for ratio in ratios]
    split_names = ("train", "valid", "test")
    assigned: dict[str, list[Any]] = {split: [] for split in split_names}
    for index, group_id in enumerate(shuffled):
        if index < 3:
            split_index = index
        else:
            split_index = max(
                range(3),
                key=lambda candidate: target[candidate] - len(assigned[split_names[candidate]]),
            )
        assigned[split_names[split_index]].extend(groups[group_id])
    annotated = {
        split: [(sample, "", "grouped") for sample in split_samples] for split, split_samples in assigned.items()
    }
    return _write_manifests(
        output,
        source,
        annotated,
        "paired-directory",
        seed,
        label_map,
        ignore_values,
        {"group_regex": str(group_regex), "group_count": len(groups)},
    )


def _group_id(sample_id: str, pattern: re.Pattern[str] | None, provided: str = "") -> str:
    if provided:
        return provided
    if pattern is None:
        return ""
    match = pattern.search(sample_id)
    if match is None:
        raise ManifestError(f"group_regex does not match sample_id: {sample_id}")
    return match.group(1) if match.lastindex else match.group(0)


def _write_rows(
    *,
    output: Path,
    source: Path,
    rows_by_split: dict[str, list[dict[str, object]]],
    provider: str,
    split_method: str,
    seed: int | None,
    label_map: dict[int, int],
    ignore_values: set[int],
    label_schema: LabelSchema | None = None,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for split in ("train", "valid", "test"):
        if not rows_by_split.get(split):
            raise ManifestError(f"provider produced an empty split: {split}")
        path = output / f"{split}.csv"
        _write_dict_rows(path, MANIFEST_FIELDS, rows_by_split[split])
        paths[split] = path
    counts = {split: len(rows) for split, rows in rows_by_split.items()}
    _write_metadata(
        output,
        source,
        provider,
        seed,
        label_map,
        ignore_values,
        paths,
        counts,
        provider_name=provider,
        split_method_name=split_method,
        label_schema=label_schema,
    )
    return paths


def prepare_split_folders_provider(**kwargs: Any) -> dict[str, Path]:
    params = dict(kwargs.pop("params", {}))
    _reject_unknown_params(params, {"group_regex"})
    kwargs.pop("split_ratios")
    source = Path(kwargs["data_dir"]).resolve()
    output = Path(kwargs["manifest_dir"]).resolve()
    pattern = re.compile(str(params["group_regex"])) if "group_regex" in params else None
    group_splits: dict[str, str] = {}
    rows_by_split: dict[str, list[dict[str, object]]] = {}
    for split in ("train", "valid", "test"):
        split_root = source / split
        samples = _validate_source(split_root, output, kwargs["label_map"], kwargs["ignore_values"])
        rows: list[dict[str, object]] = []
        for sample in samples:
            group = _group_id(sample.sample_id, pattern)
            if group and group in group_splits and group_splits[group] != split:
                raise ManifestError(f"group {group!r} appears in multiple splits")
            if group:
                group_splits[group] = split
            rows.append(
                vars(
                    ManifestRow(
                        sample_id=sample.sample_id,
                        image_path=_relative_path(sample.image_path, output),
                        mask_path=_relative_path(sample.mask_path, output),
                        split=split,
                        width=sample.width,
                        height=sample.height,
                        source_split=split,
                        group_id=group,
                    )
                )
            )
        rows_by_split[split] = rows
    return _write_rows(
        output=output,
        source=source,
        rows_by_split=rows_by_split,
        provider="split_folders",
        split_method="provided",
        seed=kwargs["seed"],
        label_map=kwargs["label_map"],
        ignore_values=kwargs["ignore_values"],
    )


def _resolve_manifest_path(value: str, manifest_path: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def prepare_manifest_provider(**kwargs: Any) -> dict[str, Path]:
    params = dict(kwargs.pop("params", {}))
    _reject_unknown_params(params, {"manifests"})
    kwargs.pop("split_ratios")
    source = Path(kwargs["data_dir"]).resolve()
    output = Path(kwargs["manifest_dir"]).resolve()
    manifests = params.get("manifests")
    if not isinstance(manifests, Mapping) or set(manifests) != {"train", "valid", "test"}:
        raise ValueError("manifest provider requires params.manifests for train, valid, and test")
    rows_by_split: dict[str, list[dict[str, object]]] = {}
    group_splits: dict[str, str] = {}
    seen_samples: set[str] = set()
    for split in ("train", "valid", "test"):
        input_path = Path(str(manifests[split])).resolve()
        if not input_path.is_file():
            raise FileNotFoundError(f"input manifest not found: {input_path}")
        with input_path.open(newline="") as handle:
            source_rows = list(csv.DictReader(handle))
        rows: list[dict[str, object]] = []
        for row in source_rows:
            if not {"sample_id", "image_path", "mask_path"} <= set(row):
                raise ManifestError(f"manifest is missing required columns: {input_path}")
            sample_id = row["sample_id"]
            if sample_id in seen_samples:
                raise ManifestError(f"sample_id appears in multiple splits: {sample_id}")
            seen_samples.add(sample_id)
            group = _group_id(sample_id, None, row.get("group_id", ""))
            if group and group in group_splits and group_splits[group] != split:
                raise ManifestError(f"group {group!r} appears in multiple splits")
            if group:
                group_splits[group] = split
            image_path = _resolve_manifest_path(row["image_path"], input_path)
            mask_path = _resolve_manifest_path(row["mask_path"], input_path)
            try:
                with Image.open(image_path) as image:
                    image.load()
                    width, height = image.size
                with Image.open(mask_path) as mask:
                    mask.load()
                    if mask.size != (width, height):
                        raise ManifestError(f"size mismatch for {sample_id}: image={(width, height)}, mask={mask.size}")
            except (OSError, UnidentifiedImageError) as exc:
                raise ManifestError(f"invalid files for {sample_id}: {exc}") from exc
            rows.append(
                vars(
                    ManifestRow(
                        sample_id=sample_id,
                        image_path=_relative_path(image_path, output),
                        mask_path=_relative_path(mask_path, output),
                        split=split,
                        width=width,
                        height=height,
                        source_split=split,
                        group_id=group,
                    )
                )
            )
        rows_by_split[split] = rows
    return _write_rows(
        output=output,
        source=source,
        rows_by_split=rows_by_split,
        provider="manifest",
        split_method="provided",
        seed=None,
        label_map=kwargs["label_map"],
        ignore_values=kwargs["ignore_values"],
    )

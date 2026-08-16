"""Dataset preparation registry and discovery metadata."""

from __future__ import annotations

import difflib
import importlib.util
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .providers import (
    prepare_manifest_provider,
    prepare_oxford_provider,
    prepare_paired_provider,
    prepare_split_folders_provider,
)


@dataclass(frozen=True)
class DatasetSpec:
    """Preparation and teaching metadata for one dataset provider."""

    name: str
    provider: str
    prepare: Callable[..., dict[str, Path]]
    dependency: str | None = None
    install_hint: str | None = None
    parameters: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetStatus:
    available: bool
    reason: str | None = None
    install_hint: str | None = None


_REGISTRY: dict[str, DatasetSpec] = {}


def register_dataset(spec: DatasetSpec) -> None:
    if spec.name in _REGISTRY:
        raise ValueError(f"dataset {spec.name!r} is already registered")
    _REGISTRY[spec.name] = spec


def available_datasets() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def get_dataset_spec(name: str) -> DatasetSpec:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        choices = available_datasets()
        suggestion = difflib.get_close_matches(name, choices, n=1)
        hint = f"; did you mean {suggestion[0]!r}?" if suggestion else ""
        raise KeyError(f"unknown dataset {name!r}{hint}; available datasets: {', '.join(choices)}") from exc


def dataset_status(name: str) -> DatasetStatus:
    spec = get_dataset_spec(name)
    if spec.dependency is None or importlib.util.find_spec(spec.dependency) is not None:
        return DatasetStatus(True, install_hint=spec.install_hint)
    return DatasetStatus(False, f"missing optional dependency: {spec.dependency}", spec.install_hint)


def prepare_dataset(
    name: str,
    *,
    data_dir: str | Path,
    manifest_dir: str | Path,
    split_ratios: list[float],
    seed: int,
    label_map: dict[int, int],
    ignore_values: set[int],
    params: dict[str, Any],
) -> dict[str, Path]:
    spec = get_dataset_spec(name)
    status = dataset_status(name)
    if not status.available:
        raise ImportError(status.reason)
    return spec.prepare(
        data_dir=data_dir,
        manifest_dir=manifest_dir,
        split_ratios=split_ratios,
        seed=seed,
        label_map=label_map,
        ignore_values=ignore_values,
        params=params,
    )


register_dataset(DatasetSpec("oxford_pet", "image-segmenter", prepare_oxford_provider))
register_dataset(
    DatasetSpec(
        "paired_masks",
        "image-segmenter",
        prepare_paired_provider,
        parameters={"group_regex": "Optional regex used to keep related samples in one split."},
    )
)
register_dataset(
    DatasetSpec(
        "split_folders",
        "image-segmenter",
        prepare_split_folders_provider,
        parameters={"group_regex": "Optional regex used to audit group separation."},
    )
)
register_dataset(
    DatasetSpec(
        "manifest",
        "image-segmenter",
        prepare_manifest_provider,
        parameters={"manifests": "Mapping with train, valid, and test CSV paths."},
    )
)

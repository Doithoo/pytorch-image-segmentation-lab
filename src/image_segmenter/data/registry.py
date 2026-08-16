"""Dataset preparation registry and discovery metadata."""

from __future__ import annotations

import difflib
import importlib.util
from collections.abc import Callable, Mapping, Sized
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import torch
from torch.utils.data import Dataset

from ..extensions import ExtensionError, load_factory
from .providers import (
    prepare_manifest_provider,
    prepare_oxford_provider,
    prepare_paired_provider,
    prepare_split_folders_provider,
)
from .schema import LabelSchema
from .voc import prepare_voc2012_provider


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


def _validate_external_sample(
    sample: object,
    *,
    split: str,
    label_schema: LabelSchema,
) -> None:
    if not isinstance(sample, tuple) or len(sample) != 3:
        raise ExtensionError(f"external dataset split {split!r} must return (image, target, sample_id)")
    image, target, sample_id = sample
    if not isinstance(image, torch.Tensor) or image.ndim != 3 or not image.is_floating_point():
        raise ExtensionError(f"external dataset split {split!r} image must be a float Tensor[C,H,W]")
    if not isinstance(target, torch.Tensor) or target.ndim != 2 or target.dtype != torch.long:
        raise ExtensionError(f"external dataset split {split!r} target must be a long Tensor[H,W]")
    if tuple(image.shape[-2:]) != tuple(target.shape):
        raise ExtensionError(f"external dataset split {split!r} image and target spatial shapes differ")
    if not isinstance(sample_id, str) or not sample_id:
        raise ExtensionError(f"external dataset split {split!r} sample_id must be a non-empty string")
    allowed = set(range(label_schema.num_classes)) | {label_schema.ignore_index}
    unexpected = sorted({int(value) for value in target.unique().tolist()} - allowed)
    if unexpected:
        raise ExtensionError(f"external dataset split {split!r} target contains unknown class ids: {unexpected}")


def load_external_datasets(
    factory_path: str,
    *,
    data_dir: str | Path,
    manifest_dir: str | Path,
    transforms: Mapping[str, Callable[..., Any]],
    label_schema: LabelSchema,
    params: dict[str, Any],
) -> dict[str, Dataset[Any]]:
    """Construct and validate train, valid, and test external Dataset objects."""
    split_names = ("train", "valid", "test")
    if set(transforms) != set(split_names):
        raise ExtensionError("external dataset transforms must define train, valid, and test")
    required = ("split", "data_dir", "manifest_dir", "transform", "label_schema", *params)
    factory = load_factory(factory_path, required_keywords=required)
    datasets: dict[str, Dataset[Any]] = {}
    for split in split_names:
        try:
            dataset = factory(
                split=split,
                data_dir=Path(data_dir),
                manifest_dir=Path(manifest_dir),
                transform=transforms[split],
                label_schema=label_schema,
                **params,
            )
        except ExtensionError:
            raise
        except Exception as exc:
            raise ExtensionError(f"external dataset factory {factory_path!r} failed for {split}: {exc}") from exc
        if not isinstance(dataset, Dataset):
            raise ExtensionError(f"external dataset factory {factory_path!r} did not return torch.utils.data.Dataset")
        if not isinstance(dataset, Sized):
            raise ExtensionError(f"external dataset split {split!r} must define __len__")
        if len(cast(Sized, dataset)) == 0:
            raise ExtensionError(f"external dataset split {split!r} contains no samples")
        _validate_external_sample(dataset[0], split=split, label_schema=label_schema)
        datasets[split] = dataset
    return datasets


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
        "voc2012",
        "image-segmenter",
        prepare_voc2012_provider,
        parameters={
            "test_policy": "Set to use_valid when validation is also the explicit evaluation split.",
            "test_list": "Optional path to a labeled evaluation sample-ID list.",
        },
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

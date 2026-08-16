"""Fast model-data contract validation before a training run creates artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .data.schema import LabelSchema


@dataclass(frozen=True)
class PreflightReport:
    """Observed shapes and values from one validated batch."""

    image_shape: tuple[int, ...]
    target_shape: tuple[int, ...]
    logits_shape: tuple[int, ...]
    target_values: tuple[int, ...]
    class_names: tuple[str, ...]


def validate_batch_contract(
    model: nn.Module,
    batch: object,
    label_schema: LabelSchema,
    *,
    expected_in_channels: int,
    device: torch.device | str = "cpu",
) -> PreflightReport:
    """Validate one real batch and model output against a LabelSchema."""
    if not isinstance(batch, list | tuple) or len(batch) != 3:
        raise ValueError("data loader batch must be (images, targets, sample_ids)")
    images, targets, _sample_ids = batch
    if not isinstance(images, torch.Tensor) or images.ndim != 4 or not images.is_floating_point():
        raise ValueError("images must be float [B,C,H,W] tensors")
    if images.shape[1] != expected_in_channels:
        raise ValueError(f"expected {expected_in_channels} image channels, got {images.shape[1]}")
    if not isinstance(targets, torch.Tensor) or targets.ndim != 3 or targets.dtype != torch.long:
        raise ValueError("targets must be long [B,H,W] tensors")
    if targets.shape[0] != images.shape[0] or targets.shape[-2:] != images.shape[-2:]:
        raise ValueError(f"target shape {tuple(targets.shape)} does not match image batch {tuple(images.shape)}")
    target_values = tuple(sorted(int(value) for value in targets.unique().tolist()))
    class_names = tuple(item.name for item in label_schema.classes)
    allowed = set(range(label_schema.num_classes)) | {label_schema.ignore_index}
    unknown = sorted(set(target_values) - allowed)
    if unknown:
        raise ValueError(
            f"target values {unknown} are outside schema classes {class_names} "
            f"and ignore_index {label_schema.ignore_index}"
        )
    if all(value == label_schema.ignore_index for value in target_values):
        raise ValueError(f"all pixels use ignore_index {label_schema.ignore_index}")

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            output: Any = model(images.to(device))
    finally:
        model.train(was_training)
    if isinstance(output, Mapping):
        raise ValueError("model returned a mapping; register an output adapter that selects logits")
    if not isinstance(output, torch.Tensor) or output.ndim != 4:
        shape = tuple(output.shape) if isinstance(output, torch.Tensor) else type(output).__name__
        raise ValueError(f"model must return logits [B,K,H,W], got {shape}")
    if output.shape[0] != images.shape[0]:
        raise ValueError(f"logits batch size {output.shape[0]} does not match images {images.shape[0]}")
    if output.shape[1] != label_schema.num_classes:
        raise ValueError(
            f"logits class channels {output.shape[1]} expected {label_schema.num_classes} "
            f"for schema {class_names}; logits shape={tuple(output.shape)}"
        )
    if output.shape[-2:] != targets.shape[-2:]:
        raise ValueError(
            f"logits spatial shape {tuple(output.shape[-2:])} does not match target "
            f"{tuple(targets.shape[-2:])}; logits shape={tuple(output.shape)}"
        )
    return PreflightReport(
        image_shape=tuple(images.shape),
        target_shape=tuple(targets.shape),
        logits_shape=tuple(output.shape),
        target_values=target_values,
        class_names=class_names,
    )

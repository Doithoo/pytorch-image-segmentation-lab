"""Metadata and output adapters used by the model registry."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol

import torch
from torch import nn


class OutputAdapter(Protocol):
    """Convert a provider-specific model output to segmentation logits."""

    def __call__(self, output: object) -> torch.Tensor: ...


@dataclass(frozen=True)
class MappingOutputAdapter:
    """Select one tensor from a mapping returned by a model provider."""

    key: str

    def __call__(self, output: object) -> torch.Tensor:
        if not isinstance(output, Mapping):
            raise TypeError(f"expected model output mapping containing {self.key!r}")
        if self.key not in output:
            raise ValueError(f"model output mapping does not contain {self.key!r}")
        value = output[self.key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"model output {self.key!r} must be a tensor")
        return value


class AdaptedModel(nn.Module):
    """Apply an output adapter while retaining an ordinary nn.Module API."""

    def __init__(self, model: nn.Module, adapter: OutputAdapter) -> None:
        super().__init__()
        self.model = model
        self.adapter = adapter

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.adapter(self.model(image))


@dataclass(frozen=True)
class ModelSpec:
    """Construction and teaching metadata for one stable model name."""

    name: str
    provider: str
    factory: Callable[..., nn.Module]
    dependency: str | None = None
    install_hint: str | None = None
    supports_pretrained: bool = False
    parameters: Mapping[str, str] = field(default_factory=dict)
    output_adapter: OutputAdapter | None = None
    input_constraints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or not self.provider:
            raise ValueError("model name and provider must be non-empty")
        if not callable(self.factory):
            raise ValueError("model factory must be callable")


@dataclass(frozen=True)
class ModelStatus:
    """Dependency availability for a registered model."""

    available: bool
    reason: str | None = None
    install_hint: str | None = None


def attach_adapter(model: nn.Module, adapter: OutputAdapter | None) -> nn.Module:
    """Wrap a model only when its provider output needs adaptation."""
    return AdaptedModel(model, adapter) if adapter is not None else model

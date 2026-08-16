"""Explicit and inspectable loading for user-provided Python factories."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, Iterable

import torch
from torch import nn


class ExtensionError(RuntimeError):
    """An external factory could not satisfy its documented contract."""


def load_factory(path: str, *, required_keywords: Iterable[str] = ()) -> Callable[..., object]:
    """Load and validate one explicit ``module:function`` reference."""
    if not isinstance(path, str) or path.count(":") != 1:
        raise ExtensionError(f"invalid extension path {path!r}; expected module:function")
    module_name, attribute_name = path.split(":", 1)
    if not module_name or not attribute_name:
        raise ExtensionError(f"invalid extension path {path!r}; expected module:function")
    try:
        module = importlib.import_module(module_name)
    except (ImportError, OSError) as exc:
        raise ExtensionError(f"could not import module {module_name!r} for {path!r}: {exc}") from exc
    try:
        factory = getattr(module, attribute_name)
    except AttributeError as exc:
        raise ExtensionError(f"module {module_name!r} has no attribute {attribute_name!r} for {path!r}") from exc
    if not callable(factory):
        raise ExtensionError(f"extension attribute {path!r} is not callable")

    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError) as exc:
        raise ExtensionError(f"could not inspect extension factory {path!r}: {exc}") from exc
    parameters = signature.parameters
    supplied = set(required_keywords)
    accepts_extra_keywords = any(item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters.values())
    for keyword in supplied:
        parameter = parameters.get(keyword)
        if parameter is None and not accepts_extra_keywords:
            raise ExtensionError(f"factory {path!r} does not accept required keyword {keyword!r}")
        if parameter is not None and parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise ExtensionError(f"factory {path!r} requires {keyword!r} as a positional argument")
    for parameter in parameters.values():
        if (
            parameter.default is inspect.Parameter.empty
            and parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            and parameter.name not in supplied
        ):
            raise ExtensionError(f"factory {path!r} has unsupported required parameter {parameter.name!r}")
    return factory


class DirectLogitsContract(nn.Module):
    """Validate an external model's tensor contract on every forward call."""

    def __init__(self, model: nn.Module, *, num_classes: int, factory_path: str) -> None:
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.factory_path = factory_path

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        output = self.model(image)
        expected = (image.shape[0], self.num_classes, image.shape[-2], image.shape[-1])
        if not isinstance(output, torch.Tensor):
            raise ExtensionError(
                f"external model {self.factory_path!r} must return a tensor; got {type(output).__name__}"
            )
        if tuple(output.shape) != expected:
            raise ExtensionError(
                f"external model {self.factory_path!r} expected [B,{self.num_classes},H,W] logits; "
                f"got {tuple(output.shape)} for input {tuple(image.shape)}"
            )
        return output

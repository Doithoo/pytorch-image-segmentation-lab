"""Extensible model construction with discoverable metadata."""

from __future__ import annotations

import difflib
import importlib.util
from functools import partial
from typing import Any

from torch import nn

from ..extensions import DirectLogitsContract, ExtensionError, load_factory
from .spec import MappingOutputAdapter, ModelSpec, ModelStatus, attach_adapter
from .torchvision_models import build_torchvision_model
from .unet import SmallUNet

_REGISTRY: dict[str, ModelSpec] = {}


def register_model(spec: ModelSpec) -> None:
    """Register one stable model name and reject accidental replacement."""
    if spec.name in _REGISTRY:
        raise ValueError(f"model {spec.name!r} is already registered")
    _REGISTRY[spec.name] = spec


def available_models(*, provider: str | None = None) -> tuple[str, ...]:
    """Return registered model names in stable sorted order."""
    return tuple(sorted(name for name, spec in _REGISTRY.items() if provider is None or spec.provider == provider))


def get_model_spec(name: str) -> ModelSpec:
    """Return metadata for a registered name with a useful typo hint."""
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        choices = available_models()
        suggestion = difflib.get_close_matches(name, choices, n=1)
        hint = f"; did you mean {suggestion[0]!r}?" if suggestion else ""
        raise KeyError(f"unknown model {name!r}{hint}; available models: {', '.join(choices)}") from exc


def model_status(name: str) -> ModelStatus:
    """Report whether the registered model's optional dependency is importable."""
    spec = get_model_spec(name)
    if spec.dependency is None or importlib.util.find_spec(spec.dependency) is not None:
        return ModelStatus(available=True, install_hint=spec.install_hint)
    return ModelStatus(
        available=False,
        reason=f"missing optional dependency: {spec.dependency}",
        install_hint=spec.install_hint,
    )


def create_model(
    name: str,
    *,
    factory: str | None = None,
    in_channels: int,
    num_classes: int,
    pretrained: bool,
    params: dict[str, Any] | None = None,
    base_channels: int | None = None,
) -> nn.Module:
    """Construct a registered model that follows the direct-logits contract."""
    if factory is not None:
        resolved_params = dict(params or {})
        if base_channels is not None:
            resolved_params.setdefault("base_channels", base_channels)
        supplied = ("in_channels", "num_classes", "pretrained", *resolved_params)
        external_factory = load_factory(factory, required_keywords=supplied)
        try:
            external_model = external_factory(
                in_channels=in_channels,
                num_classes=num_classes,
                pretrained=pretrained,
                **resolved_params,
            )
        except ExtensionError:
            raise
        except Exception as exc:
            raise ExtensionError(f"external model factory {factory!r} failed: {exc}") from exc
        if not isinstance(external_model, nn.Module):
            raise ExtensionError(f"external model factory {factory!r} did not return torch.nn.Module")
        contracted = DirectLogitsContract(external_model, num_classes=num_classes, factory_path=factory)
        dynamic_contracted: Any = contracted
        dynamic_contracted.checkpoint_metadata = {
            "architecture": name,
            "provider": "external",
            "factory": factory,
            "parameters": resolved_params,
            "pretrained": pretrained,
        }
        return contracted
    spec = get_model_spec(name)
    status = model_status(name)
    if not status.available:
        detail = f"; {status.install_hint}" if status.install_hint else ""
        raise ImportError(f"{status.reason}{detail}")
    if pretrained and not spec.supports_pretrained:
        raise ValueError(f"{name} does not provide pretrained weights")
    resolved_params = dict(params or {})
    if base_channels is not None and "base_channels" in spec.parameters:
        if "base_channels" in resolved_params and resolved_params["base_channels"] != base_channels:
            raise ValueError("base_channels conflicts with params.base_channels")
        resolved_params["base_channels"] = base_channels
    model = spec.factory(
        in_channels=in_channels,
        num_classes=num_classes,
        pretrained=pretrained,
        **resolved_params,
    )
    if not isinstance(model, nn.Module):
        raise TypeError(f"model factory for {name!r} did not return torch.nn.Module")
    adapted = attach_adapter(model, spec.output_adapter)
    provider_metadata = dict(getattr(model, "checkpoint_metadata", {}))
    dynamic_adapted: Any = adapted
    dynamic_adapted.checkpoint_metadata = {
        **provider_metadata,
        "architecture": name,
        "provider": spec.provider,
        "parameters": resolved_params,
    }
    return adapted


def _create_unet(
    *,
    in_channels: int,
    num_classes: int,
    pretrained: bool,
    base_channels: int = 16,
) -> nn.Module:
    if pretrained:
        raise ValueError("unet_small does not provide pretrained weights")
    return SmallUNet(in_channels=in_channels, num_classes=num_classes, base_channels=base_channels)


register_model(
    ModelSpec(
        name="unet_small",
        provider="image-segmenter",
        factory=_create_unet,
        supports_pretrained=False,
        parameters={"base_channels": "Width of the first encoder stage; later stages scale from it."},
        input_constraints=("Height and width must be divisible by 16.",),
    )
)

for _name in (
    "fcn_resnet50",
    "deeplabv3_resnet50",
    "deeplabv3_mobilenet_v3_large",
    "lraspp_mobilenet_v3_large",
):
    register_model(
        ModelSpec(
            name=_name,
            provider="torchvision",
            factory=partial(build_torchvision_model, _name),
            dependency="torchvision",
            supports_pretrained=True,
            output_adapter=MappingOutputAdapter("out"),
            input_constraints=("Requires three-channel RGB input.",),
        )
    )

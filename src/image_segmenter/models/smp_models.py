"""Lazy constructors for optional segmentation-models-pytorch models."""

from __future__ import annotations

from typing import Any

from torch import nn

_ARCHITECTURES = {
    "smp_unet": "Unet",
    "smp_unetplusplus": "UnetPlusPlus",
    "smp_fpn": "FPN",
    "smp_pan": "PAN",
}


def build_smp_model(
    architecture: str,
    *,
    in_channels: int,
    num_classes: int,
    pretrained: bool,
    encoder_name: str = "resnet18",
    encoder_weights: str | None = None,
    **params: Any,
) -> nn.Module:
    """Build an SMP model while keeping the optional import out of core startup."""
    import segmentation_models_pytorch as smp

    if architecture not in _ARCHITECTURES:
        raise ValueError(f"unsupported SMP architecture: {architecture}")
    if in_channels <= 0 or num_classes <= 0:
        raise ValueError("SMP input and class channel counts must be positive")
    resolved_weights = encoder_weights
    if pretrained and resolved_weights is None:
        resolved_weights = "imagenet"
    constructor = getattr(smp, _ARCHITECTURES[architecture])
    model = constructor(
        encoder_name=encoder_name,
        encoder_weights=resolved_weights,
        in_channels=in_channels,
        classes=num_classes,
        **params,
    )
    dynamic_model: Any = model
    dynamic_model.checkpoint_metadata = {
        "architecture": architecture,
        "encoder_name": encoder_name,
        "pretrained_weights": resolved_weights,
    }
    return model

"""Offline-safe construction helpers for maintained torchvision models."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torch import nn
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    DeepLabV3_ResNet50_Weights,
    FCN_ResNet50_Weights,
    LRASPP_MobileNet_V3_Large_Weights,
    deeplabv3_mobilenet_v3_large,
    deeplabv3_resnet50,
    fcn_resnet50,
    lraspp_mobilenet_v3_large,
)

_BUILDERS: dict[str, Callable[..., nn.Module]] = {
    "fcn_resnet50": fcn_resnet50,
    "deeplabv3_resnet50": deeplabv3_resnet50,
    "deeplabv3_mobilenet_v3_large": deeplabv3_mobilenet_v3_large,
    "lraspp_mobilenet_v3_large": lraspp_mobilenet_v3_large,
}
_DEFAULT_WEIGHTS: dict[str, Any] = {
    "fcn_resnet50": FCN_ResNet50_Weights.DEFAULT,
    "deeplabv3_resnet50": DeepLabV3_ResNet50_Weights.DEFAULT,
    "deeplabv3_mobilenet_v3_large": DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT,
    "lraspp_mobilenet_v3_large": LRASPP_MobileNet_V3_Large_Weights.DEFAULT,
}


def _replace_pretrained_head(name: str, model: nn.Module, num_classes: int) -> None:
    dynamic_model: Any = model
    classifier: Any = dynamic_model.classifier
    if name == "lraspp_mobilenet_v3_large":
        low = classifier.low_classifier
        high = classifier.high_classifier
        classifier.low_classifier = nn.Conv2d(low.in_channels, num_classes, kernel_size=1)
        classifier.high_classifier = nn.Conv2d(high.in_channels, num_classes, kernel_size=1)
        return
    final = classifier[-1]
    classifier[-1] = nn.Conv2d(final.in_channels, num_classes, kernel_size=1)
    if getattr(model, "aux_classifier", None) is not None:
        dynamic_model.aux_classifier = None


def build_torchvision_model(
    architecture: str,
    *,
    in_channels: int,
    num_classes: int,
    pretrained: bool,
    **params: Any,
) -> nn.Module:
    """Build a maintained torchvision segmenter without implicit downloads."""
    if in_channels != 3:
        raise ValueError(f"{architecture} requires three input channels")
    if params:
        unknown = ", ".join(sorted(params))
        raise ValueError(f"unsupported parameters for {architecture}: {unknown}")
    builder = _BUILDERS[architecture]
    weights = _DEFAULT_WEIGHTS[architecture] if pretrained else None
    if weights is None:
        kwargs: dict[str, Any] = {
            "weights": None,
            "weights_backbone": None,
            "num_classes": num_classes,
        }
        if architecture != "lraspp_mobilenet_v3_large":
            kwargs["aux_loss"] = False
        model = builder(**kwargs)
    else:
        model = builder(weights=weights)
        _replace_pretrained_head(architecture, model, num_classes)
    dynamic_model: Any = model
    dynamic_model.checkpoint_metadata = {
        "architecture": architecture,
        "pretrained_weights": weights.name if weights is not None else None,
        "classification_head_initialized": "random" if weights is not None else None,
    }
    return model

"""Small explicit model registry."""

from __future__ import annotations

from torch import nn

from .unet import SmallUNet


def available_models() -> tuple[str, ...]:
    """Return stable model names exposed by the current release."""
    return ("unet_small",)


def create_model(
    name: str,
    *,
    in_channels: int,
    num_classes: int,
    base_channels: int,
    pretrained: bool,
) -> nn.Module:
    """Construct a model that follows the direct-logits contract."""
    if name != "unet_small":
        choices = ", ".join(available_models())
        raise KeyError(f"unknown model {name!r}; available models: {choices}")
    if pretrained:
        raise ValueError("unet_small does not provide pretrained weights")
    return SmallUNet(in_channels=in_channels, num_classes=num_classes, base_channels=base_channels)

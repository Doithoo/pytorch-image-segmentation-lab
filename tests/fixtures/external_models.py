from typing import Any

import torch
from torch import nn

NOT_CALLABLE = 42


def create_model(
    *,
    in_channels: int,
    num_classes: int,
    pretrained: bool,
    width: int = 4,
) -> nn.Module:
    del pretrained, width
    return nn.Conv2d(in_channels, num_classes, kernel_size=1)


def requires_unknown(
    *,
    in_channels: int,
    num_classes: int,
    pretrained: bool,
    secret: str,
) -> nn.Module:
    del in_channels, num_classes, pretrained, secret
    return nn.Identity()


def returns_object(**_: Any) -> object:
    return object()


class WrongChannels(nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return image[:, :1]


def wrong_shape(**_: Any) -> nn.Module:
    return WrongChannels()

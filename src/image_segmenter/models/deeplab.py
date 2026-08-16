"""Torchvision DeepLabV3 adapter for the shared direct-logits contract."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50


class DeepLabV3Adapter(nn.Module):
    """Expose torchvision DeepLabV3 output as a logits tensor."""

    def __init__(self, num_classes: int = 2, *, pretrained: bool = True) -> None:
        super().__init__()
        weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
        if weights is None:
            model = deeplabv3_resnet50(weights=None, weights_backbone=None, aux_loss=False)
        else:
            model = deeplabv3_resnet50(weights=weights)
        classifier_channels = model.classifier[-1].in_channels
        model.classifier[-1] = nn.Conv2d(classifier_channels, num_classes, kernel_size=1)
        model.aux_classifier = None
        self.model = model
        self.checkpoint_metadata = {
            "architecture": "deeplabv3_resnet50",
            "pretrained_weights": weights.name if weights is not None else None,
        }

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)["out"]

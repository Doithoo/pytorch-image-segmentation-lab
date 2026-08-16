"""Ignore-aware losses for binary semantic segmentation."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def foreground_soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, ignore_index: int) -> torch.Tensor:
    """Compute soft Dice loss for class 1 over valid pixels."""
    if logits.ndim != 4 or logits.shape[1] != 2:
        raise ValueError(f"expected logits [B,2,H,W], got {tuple(logits.shape)}")
    if target.shape != logits.shape[:1] + logits.shape[2:]:
        raise ValueError(f"target shape {tuple(target.shape)} does not match logits {tuple(logits.shape)}")
    valid = target != ignore_index
    if not valid.any():
        raise ValueError("batch has no valid pixels")
    probability = logits.softmax(dim=1)[:, 1][valid]
    foreground = (target[valid] == 1).to(probability.dtype)
    intersection = (probability * foreground).sum()
    denominator = probability.sum() + foreground.sum()
    return 1.0 - (2.0 * intersection + 1e-6) / (denominator + 1e-6)


class SegmentationLoss(nn.Module):
    """Weighted cross-entropy and foreground Dice composition."""

    def __init__(self, cross_entropy_weight: float, dice_weight: float, ignore_index: int) -> None:
        super().__init__()
        if cross_entropy_weight < 0 or dice_weight < 0 or cross_entropy_weight + dice_weight <= 0:
            raise ValueError("loss weights must be non-negative with a positive total")
        self.cross_entropy_weight = cross_entropy_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        valid = target != self.ignore_index
        if not valid.any():
            raise ValueError("batch has no valid pixels")
        cross_entropy = F.cross_entropy(logits, target, ignore_index=self.ignore_index)
        has_foreground = bool((target[valid] == 1).any().item())
        if self.dice_weight > 0 and has_foreground:
            dice = foreground_soft_dice_loss(logits, target, self.ignore_index)
            dice_active = logits.new_tensor(1.0)
        else:
            dice = logits.sum() * 0.0
            dice_active = logits.new_tensor(0.0)
        total = self.cross_entropy_weight * cross_entropy + self.dice_weight * dice
        components = {
            "cross_entropy": cross_entropy.detach(),
            "dice": dice.detach(),
            "dice_active": dice_active.detach(),
        }
        return total, components

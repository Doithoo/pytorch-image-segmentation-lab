"""Ignore-aware losses for multiclass semantic segmentation."""

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def multiclass_soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
    *,
    class_indices: Sequence[int],
) -> torch.Tensor:
    """Average soft Dice loss over selected classes present in the target."""
    if logits.ndim != 4 or logits.shape[1] < 2:
        raise ValueError(f"expected logits [B,K,H,W] with K >= 2, got {tuple(logits.shape)}")
    if target.shape != logits.shape[:1] + logits.shape[2:]:
        raise ValueError(f"target shape {tuple(target.shape)} does not match logits {tuple(logits.shape)}")
    valid = target != ignore_index
    if not valid.any():
        raise ValueError("batch has no valid pixels")
    selected = tuple(class_indices)
    if not selected or any(index < 0 or index >= logits.shape[1] for index in selected):
        raise ValueError("dice class_indices must select valid model classes")
    probabilities = logits.softmax(dim=1)
    losses: list[torch.Tensor] = []
    for class_index in selected:
        target_class = (target == class_index) & valid
        if not target_class.any():
            continue
        probability = probabilities[:, class_index][valid]
        expected = target_class[valid].to(probability.dtype)
        intersection = (probability * expected).sum()
        denominator = probability.sum() + expected.sum()
        losses.append(1.0 - (2.0 * intersection + 1e-6) / (denominator + 1e-6))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def foreground_soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, ignore_index: int) -> torch.Tensor:
    """Compute soft Dice loss for class 1 over valid pixels."""
    if logits.ndim != 4 or logits.shape[1] != 2:
        raise ValueError(f"expected logits [B,2,H,W], got {tuple(logits.shape)}")
    return multiclass_soft_dice_loss(logits, target, ignore_index, class_indices=(1,))


class SegmentationLoss(nn.Module):
    """Weighted cross-entropy and foreground Dice composition."""

    def __init__(
        self,
        cross_entropy_weight: float,
        dice_weight: float,
        ignore_index: int,
        dice_classes: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if cross_entropy_weight < 0 or dice_weight < 0 or cross_entropy_weight + dice_weight <= 0:
            raise ValueError("loss weights must be non-negative with a positive total")
        self.cross_entropy_weight = cross_entropy_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index
        self.dice_classes = tuple(dice_classes) if dice_classes is not None else None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        valid = target != self.ignore_index
        if not valid.any():
            raise ValueError("batch has no valid pixels")
        cross_entropy = F.cross_entropy(logits, target, ignore_index=self.ignore_index)
        selected = self.dice_classes or tuple(range(1, logits.shape[1]))
        has_selected_class = any(bool((target[valid] == index).any().item()) for index in selected)
        if self.dice_weight > 0 and has_selected_class:
            dice = multiclass_soft_dice_loss(
                logits,
                target,
                self.ignore_index,
                class_indices=selected,
            )
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

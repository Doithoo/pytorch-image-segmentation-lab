"""Streaming confusion-matrix metrics for semantic segmentation."""

from __future__ import annotations

import math

import torch


class SegmentationMetrics:
    """Accumulate predictions without retaining full-resolution batches."""

    def __init__(self, num_classes: int = 2, ignore_index: int = 255) -> None:
        if num_classes != 2:
            raise ValueError("this release supports exactly two classes")
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        """Add logits or class-index predictions for one batch."""
        if prediction.ndim == 4:
            prediction = prediction.argmax(dim=1)
        if prediction.shape != target.shape:
            raise ValueError(f"prediction shape {tuple(prediction.shape)} does not match target {tuple(target.shape)}")
        valid = target != self.ignore_index
        if not valid.any():
            return
        prediction_valid = prediction[valid].to(torch.int64)
        target_valid = target[valid].to(torch.int64)
        if (prediction_valid < 0).any() or (prediction_valid >= self.num_classes).any():
            raise ValueError("prediction contains an invalid class index")
        if (target_valid < 0).any() or (target_valid >= self.num_classes).any():
            raise ValueError("target contains an invalid class index")
        encoded = target_valid * self.num_classes + prediction_valid
        counts = torch.bincount(encoded, minlength=self.num_classes**2)
        self.confusion += counts.reshape(self.num_classes, self.num_classes).cpu()

    def compute(self) -> dict[str, float | int]:
        """Return class and macro scores from accumulated counts."""
        valid_pixels = int(self.confusion.sum().item())
        if valid_pixels == 0:
            raise ValueError("metrics contain no valid pixels")
        counts = self.confusion.to(torch.float64)
        intersection = counts.diag()
        target_count = counts.sum(dim=1)
        prediction_count = counts.sum(dim=0)
        union = target_count + prediction_count - intersection
        iou = torch.where(union > 0, intersection / union, torch.nan)
        dice_denominator = target_count + prediction_count
        dice = torch.where(dice_denominator > 0, 2.0 * intersection / dice_denominator, torch.nan)
        accuracy = intersection.sum() / counts.sum()

        def value(tensor: torch.Tensor) -> float:
            return float(tensor.item())

        return {
            "background_iou": value(iou[0]),
            "pet_iou": value(iou[1]),
            "mean_iou": value(torch.nanmean(iou)),
            "pet_dice": value(dice[1]),
            "macro_dice": value(torch.nanmean(dice)),
            "pixel_accuracy": value(accuracy),
            "valid_pixels": valid_pixels,
        }

    def reset(self) -> None:
        """Clear all accumulated counts."""
        self.confusion.zero_()


def score_or_none(value: float) -> float | None:
    """Convert NaN to a JSON-friendly missing value."""
    return None if math.isnan(value) else value

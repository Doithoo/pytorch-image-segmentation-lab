"""Streaming confusion-matrix metrics for semantic segmentation."""

from __future__ import annotations

import math
from typing import Any

import torch


def _default_names(num_classes: int) -> tuple[str, ...]:
    if num_classes == 2:
        return "background", "pet"
    return tuple(f"class_{index}" for index in range(num_classes))


class SegmentationMetrics:
    """Accumulate predictions without retaining full-resolution batches."""

    def __init__(
        self,
        num_classes: int = 2,
        ignore_index: int = 255,
        *,
        class_names: tuple[str, ...] | None = None,
    ) -> None:
        if not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes < 2:
            raise ValueError("num_classes must be an integer of at least 2")
        names = class_names or _default_names(num_classes)
        if len(names) != num_classes or len(set(names)) != len(names) or any(not name for name in names):
            raise ValueError("class_names must contain one unique non-empty name per class")
        self.num_classes = num_classes
        self.class_names = tuple(names)
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
        """Return per-class and macro scores from accumulated counts."""
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
        result: dict[str, float | int] = {
            "mean_iou": float(torch.nanmean(iou).item()),
            "mean_dice": float(torch.nanmean(dice).item()),
            "macro_dice": float(torch.nanmean(dice).item()),
            "pixel_accuracy": float((intersection.sum() / counts.sum()).item()),
            "valid_pixels": valid_pixels,
        }
        for index, name in enumerate(self.class_names):
            result[f"class_iou:{name}"] = float(iou[index].item())
            result[f"class_dice:{name}"] = float(dice[index].item())
        if self.num_classes == 2 and self.class_names == ("background", "pet"):
            result["background_iou"] = result["class_iou:background"]
            result["pet_iou"] = result["class_iou:pet"]
            result["pet_dice"] = result["class_dice:pet"]
        return result

    def reset(self) -> None:
        self.confusion.zero_()


def per_image_scores(
    prediction: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
    num_classes: int = 2,
    class_names: tuple[str, ...] | None = None,
) -> list[dict[str, float | int]]:
    """Calculate class and macro metrics independently for each batch item."""
    if prediction.ndim == 4:
        prediction = prediction.argmax(dim=1)
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must have matching [B,H,W] shapes")
    names = class_names or _default_names(num_classes)
    results: list[dict[str, float | int]] = []
    for predicted_mask, target_mask in zip(prediction, target, strict=True):
        valid = target_mask != ignore_index
        valid_pixels = int(valid.sum().item())
        ignored_pixels = int((~valid).sum().item())
        row: dict[str, float | int] = {"valid_pixels": valid_pixels, "ignored_pixels": ignored_pixels}
        ious: list[float] = []
        dices: list[float] = []
        for class_id, name in enumerate(names):
            predicted_class = (predicted_mask == class_id) & valid
            target_class = (target_mask == class_id) & valid
            intersection = int((predicted_class & target_class).sum().item())
            union = int((predicted_class | target_class).sum().item())
            denominator = int(predicted_class.sum().item() + target_class.sum().item())
            iou = intersection / union if union else math.nan
            dice = 2 * intersection / denominator if denominator else math.nan
            row[f"class_iou:{name}"] = iou
            row[f"class_dice:{name}"] = dice
            ious.append(iou)
            dices.append(dice)
        finite_ious = [value for value in ious if math.isfinite(value)]
        finite_dices = [value for value in dices if math.isfinite(value)]
        row["mean_iou"] = sum(finite_ious) / len(finite_ious) if finite_ious else math.nan
        row["mean_dice"] = sum(finite_dices) / len(finite_dices) if finite_dices else math.nan
        if num_classes == 2 and tuple(names) == ("background", "pet"):
            row["pet_iou"] = row["class_iou:pet"]
            row["pet_dice"] = row["class_dice:pet"]
            target_count = int(((target_mask == 1) & valid).sum().item())
            predicted_count = int(((predicted_mask == 1) & valid).sum().item())
            row["target_foreground_ratio"] = target_count / valid_pixels if valid_pixels else math.nan
            row["predicted_foreground_ratio"] = predicted_count / valid_pixels if valid_pixels else math.nan
        results.append(row)
    return results


def serializable_metrics(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: serializable_metrics(item) for key, item in value.items()}
    if isinstance(value, list):
        return [serializable_metrics(item) for item in value]
    if isinstance(value, tuple):
        return [serializable_metrics(item) for item in value]
    return value


def score_or_none(value: float) -> float | None:
    return None if math.isnan(value) else value

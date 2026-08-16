"""Evaluation public API."""

from .metrics import SegmentationMetrics, per_image_scores, score_or_none, serializable_metrics
from .visualization import (
    colorize_mask,
    comparison_grid,
    error_overlay,
    prediction_overlay,
    save_prediction_outputs,
)

__all__ = [
    "SegmentationMetrics",
    "colorize_mask",
    "comparison_grid",
    "error_overlay",
    "per_image_scores",
    "prediction_overlay",
    "save_prediction_outputs",
    "score_or_none",
    "serializable_metrics",
]

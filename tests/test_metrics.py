import math

import pytest
import torch

from image_segmenter.evaluation.metrics import SegmentationMetrics


def test_metrics_ignore_255_and_report_foreground_iou() -> None:
    metrics = SegmentationMetrics(num_classes=2, ignore_index=255)
    prediction = torch.tensor([[[0, 1], [1, 0]]])
    target = torch.tensor([[[0, 1], [0, 255]]])
    metrics.update(prediction, target)
    result = metrics.compute()
    assert result["valid_pixels"] == 3
    assert result["pet_iou"] == pytest.approx(0.5)
    assert result["background_iou"] == pytest.approx(0.5)
    assert result["mean_iou"] == pytest.approx(0.5)
    assert result["pet_dice"] == pytest.approx(2 / 3)
    assert result["pixel_accuracy"] == pytest.approx(2 / 3)


def test_absent_class_is_nan_and_excluded_from_macro() -> None:
    metrics = SegmentationMetrics(num_classes=2, ignore_index=255)
    metrics.update(torch.zeros(1, 2, 2, dtype=torch.int64), torch.zeros(1, 2, 2, dtype=torch.int64))
    result = metrics.compute()
    assert math.isnan(result["pet_iou"])
    assert result["mean_iou"] == 1.0
    assert result["macro_dice"] == 1.0

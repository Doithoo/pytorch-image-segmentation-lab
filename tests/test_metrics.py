import math

import pytest
import torch

from image_segmenter.evaluation.metrics import SegmentationMetrics, per_image_scores, serializable_metrics


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


def test_per_image_scores_separate_errors_and_ignored_pixels() -> None:
    prediction = torch.tensor([[[1, 1], [0, 1]]])
    target = torch.tensor([[[1, 0], [0, 255]]])
    result = per_image_scores(prediction, target, ignore_index=255)[0]
    assert result["pet_iou"] == pytest.approx(0.5)
    assert result["pet_dice"] == pytest.approx(2 / 3)
    assert result["target_foreground_ratio"] == pytest.approx(1 / 3)
    assert result["predicted_foreground_ratio"] == pytest.approx(2 / 3)
    assert result["valid_pixels"] == 3
    assert result["ignored_pixels"] == 1


def test_per_image_empty_foreground_is_nan() -> None:
    scores = per_image_scores(torch.zeros(1, 2, 2, dtype=torch.int64), torch.zeros(1, 2, 2, dtype=torch.int64), 255)
    assert math.isnan(scores[0]["pet_iou"])
    assert math.isnan(scores[0]["pet_dice"])


def test_serializable_metrics_converts_nested_nan_to_none() -> None:
    result = serializable_metrics({"score": math.nan, "nested": [1, {"value": 0.5}]})
    assert result == {"score": None, "nested": [1, {"value": 0.5}]}

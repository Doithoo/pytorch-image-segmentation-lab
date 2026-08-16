from pathlib import Path

import numpy as np
from PIL import Image

from image_segmenter.evaluation.visualization import (
    colorize_mask,
    comparison_grid,
    error_overlay,
    save_prediction_outputs,
)


def _example() -> tuple[Image.Image, np.ndarray, np.ndarray]:
    image = Image.new("RGB", (4, 4), color=(100, 100, 100))
    target = np.array(
        [[1, 0, 1, 255], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        dtype=np.uint8,
    )
    prediction = np.array(
        [[1, 1, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        dtype=np.uint8,
    )
    return image, target, prediction


def test_error_overlay_distinguishes_all_pixel_states() -> None:
    image, target, prediction = _example()
    rendered = np.asarray(error_overlay(image, target, prediction, ignore_index=255, alpha=1.0))
    assert len({tuple(rendered[0, index]) for index in range(4)}) == 4


def test_color_mask_and_grid_have_stable_modes_and_sizes() -> None:
    image, target, prediction = _example()
    colored = colorize_mask(prediction)
    grid = comparison_grid(image, target, prediction, ignore_index=255)
    assert colored.mode == "RGB"
    assert colored.size == image.size
    assert grid.mode == "RGB"
    assert grid.size == (image.width * 4, image.height + 24)


def test_saved_prediction_keeps_raw_class_indices(tmp_path: Path) -> None:
    image, _target, prediction = _example()
    paths = save_prediction_outputs(tmp_path / "prediction.png", image, prediction)
    assert set(np.unique(np.asarray(Image.open(paths["raw"])))) == {0, 1}
    assert Image.open(paths["color"]).mode == "RGB"
    assert Image.open(paths["overlay"]).size == image.size

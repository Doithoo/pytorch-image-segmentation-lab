from pathlib import Path

import numpy as np
import torch
from PIL import Image

from image_segmenter.data.transforms import ImageTransform
from image_segmenter.inference import Predictor


def test_image_transform_returns_network_tensor() -> None:
    transform = ImageTransform(16, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    tensor = transform(Image.new("RGB", (9, 13), color=(255, 0, 0)))
    assert tensor.shape == (3, 16, 16)
    assert tensor.dtype == torch.float32


def test_predictor_restores_original_size_and_writes_distinct_outputs(
    trained_run: dict[str, Path], tmp_path: Path
) -> None:
    predictor = Predictor(trained_run["checkpoint"], device="cpu")
    paths = predictor.predict_path(trained_run["image"], tmp_path / "prediction.png")
    raw = Image.open(paths["raw"])
    assert raw.size == Image.open(trained_run["image"]).size
    assert set(np.unique(np.asarray(raw))) <= {0, 1}
    assert paths["color"].name == "prediction.color.png"
    assert paths["overlay"].name == "prediction.overlay.png"


def test_predictor_rejects_overwriting_input(trained_run: dict[str, Path]) -> None:
    predictor = Predictor(trained_run["checkpoint"], device="cpu")
    try:
        predictor.predict_path(trained_run["image"], trained_run["image"])
    except ValueError as exc:
        assert "overwrite input" in str(exc)
    else:
        raise AssertionError("expected overwrite protection")

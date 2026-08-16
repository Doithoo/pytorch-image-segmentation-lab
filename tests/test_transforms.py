import random

import numpy as np
import pytest
from PIL import Image

from image_segmenter.config import config_from_dict
from image_segmenter.data.transforms import ImageTransform, PairedTransform, build_transform


def _image_and_mask(size: tuple[int, int] = (20, 10)) -> tuple[Image.Image, Image.Image]:
    width, height = size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(width, dtype=np.uint8)
    mask = np.tile(np.arange(width, dtype=np.uint8), (height, 1))
    return Image.fromarray(image), Image.fromarray(mask)


@pytest.mark.parametrize(("size", "expected"), [(16, (16, 16)), ([16, 32], (16, 32))])
def test_stretch_accepts_square_or_height_width_size(size: int | list[int], expected: tuple[int, int]) -> None:
    transform = PairedTransform(size, [0.0] * 3, [1.0] * 3)

    image, mask = transform(*_image_and_mask())

    assert tuple(image.shape[-2:]) == expected
    assert tuple(mask.shape) == expected


def test_letterbox_preserves_aspect_and_pads_mask_with_ignore_index() -> None:
    image = Image.new("RGB", (20, 10), color=(255, 0, 0))
    mask = Image.new("L", (20, 10), color=1)
    transform = PairedTransform(
        [20, 20],
        [0.0] * 3,
        [1.0] * 3,
        resize_mode="letterbox",
        ignore_index=255,
    )

    image_tensor, mask_tensor = transform(image, mask)

    assert tuple(image_tensor.shape) == (3, 20, 20)
    assert set(mask_tensor[:5].unique().tolist()) == {255}
    assert set(mask_tensor[5:15].unique().tolist()) == {1}
    assert set(mask_tensor[15:].unique().tolist()) == {255}


def test_crop_and_flip_sample_geometry_once_for_image_and_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    image, mask = _image_and_mask((16, 16))
    positions = iter((2, 3))
    monkeypatch.setattr(random, "randint", lambda _start, _end: next(positions))
    monkeypatch.setattr(random, "random", lambda: 0.0)
    transform = PairedTransform(
        16,
        [0.0] * 3,
        [1.0] * 3,
        horizontal_flip=1.0,
        crop_size=[8, 8],
    )

    image_tensor, mask_tensor = transform(image, mask)

    assert tuple(image_tensor.shape[-2:]) == (8, 8)
    assert np.array_equal(
        np.rint(image_tensor[0].numpy() * 255).astype(np.int64),
        mask_tensor.numpy(),
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"image_size": 0},
        {"image_size": [16, 0]},
        {"image_size": [16]},
        {"image_size": 16, "crop_size": [32, 8]},
        {"image_size": 16, "resize_mode": "unknown"},
    ],
)
def test_transform_rejects_invalid_geometry_before_loading_data(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        PairedTransform(mean=[0.0] * 3, std=[1.0] * 3, **kwargs)  # type: ignore[arg-type]


def test_configured_geometry_is_passed_to_training_transform() -> None:
    config = config_from_dict(
        {
            "config_version": 2,
            "data": {
                "image_size": [32, 48],
                "resize_mode": "letterbox",
                "crop_size": [16, 32],
            },
        }
    )

    transform = build_transform(config.data, training=True)

    assert transform.output_size == (32, 48)
    assert transform.crop_size == (16, 32)
    assert transform.resize_mode == "letterbox"


def test_image_transform_accepts_rectangular_size() -> None:
    transformed = ImageTransform([16, 32], [0.0] * 3, [1.0] * 3)(Image.new("RGB", (4, 8)))
    assert tuple(transformed.shape) == (3, 16, 32)

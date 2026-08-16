"""Pillow-based segmentation masks, overlays, and comparison grids."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

BACKGROUND_COLOR = (32, 32, 32)
PET_COLOR = (46, 204, 113)
IGNORE_COLOR = (160, 160, 160)
TRUE_POSITIVE_COLOR = (46, 204, 113)
FALSE_POSITIVE_COLOR = (231, 76, 60)
FALSE_NEGATIVE_COLOR = (52, 152, 219)
IGNORED_ERROR_COLOR = (241, 196, 15)


def _mask_array(mask: np.ndarray | Image.Image) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"mask must be two-dimensional, got {array.shape}")
    return array


def colorize_mask(mask: np.ndarray | Image.Image, ignore_index: int = 255) -> Image.Image:
    """Render class indices with a stable, documented palette."""
    array = _mask_array(mask)
    unexpected = sorted({int(value) for value in np.unique(array)} - {0, 1, ignore_index})
    if unexpected:
        raise ValueError(f"mask contains unsupported values: {unexpected}")
    colored = np.empty((*array.shape, 3), dtype=np.uint8)
    colored[array == 0] = BACKGROUND_COLOR
    colored[array == 1] = PET_COLOR
    colored[array == ignore_index] = IGNORE_COLOR
    return Image.fromarray(colored)


def _blend_states(image: Image.Image, states: np.ndarray, colors: dict[int, tuple[int, int, int]], alpha: float) -> Image.Image:
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    original = np.asarray(image.convert("RGB"), dtype=np.float32)
    if states.shape != original.shape[:2]:
        raise ValueError("image and mask sizes must match")
    rendered = original.copy()
    for state, color in colors.items():
        selected = states == state
        rendered[selected] = (1.0 - alpha) * original[selected] + alpha * np.asarray(color, dtype=np.float32)
    return Image.fromarray(np.clip(rendered, 0, 255).astype(np.uint8))


def prediction_overlay(image: Image.Image, prediction: np.ndarray | Image.Image, alpha: float = 0.45) -> Image.Image:
    """Overlay predicted foreground while leaving background unchanged."""
    array = _mask_array(prediction)
    unexpected = sorted({int(value) for value in np.unique(array)} - {0, 1})
    if unexpected:
        raise ValueError(f"prediction contains unsupported values: {unexpected}")
    return _blend_states(image, array, {1: PET_COLOR}, alpha)


def error_overlay(
    image: Image.Image,
    target: np.ndarray | Image.Image,
    prediction: np.ndarray | Image.Image,
    ignore_index: int = 255,
    alpha: float = 0.65,
) -> Image.Image:
    """Overlay true positive, false positive, false negative, and ignored pixels."""
    target_array = _mask_array(target)
    prediction_array = _mask_array(prediction)
    if target_array.shape != prediction_array.shape:
        raise ValueError("target and prediction sizes must match")
    states = np.zeros(target_array.shape, dtype=np.uint8)
    valid = target_array != ignore_index
    states[valid & (target_array == 1) & (prediction_array == 1)] = 1
    states[valid & (target_array == 0) & (prediction_array == 1)] = 2
    states[valid & (target_array == 1) & (prediction_array == 0)] = 3
    states[~valid] = 4
    return _blend_states(
        image,
        states,
        {
            1: TRUE_POSITIVE_COLOR,
            2: FALSE_POSITIVE_COLOR,
            3: FALSE_NEGATIVE_COLOR,
            4: IGNORED_ERROR_COLOR,
        },
        alpha,
    )


def comparison_grid(
    image: Image.Image,
    target: np.ndarray | Image.Image,
    prediction: np.ndarray | Image.Image,
    ignore_index: int = 255,
) -> Image.Image:
    """Create original, target, prediction, and error panels with labels."""
    original = image.convert("RGB")
    target_panel = colorize_mask(target, ignore_index)
    prediction_panel = colorize_mask(prediction, ignore_index)
    error_panel = error_overlay(original, target, prediction, ignore_index)
    panels = (original, target_panel, prediction_panel, error_panel)
    labels = ("Image", "Target", "Prediction", "Errors")
    canvas = Image.new("RGB", (original.width * 4, original.height + 24), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for index, (panel, label) in enumerate(zip(panels, labels, strict=True)):
        if panel.size != original.size:
            panel = panel.resize(original.size, resample=Image.Resampling.NEAREST)
        offset = index * original.width
        draw.text((offset + 4, 5), label, fill=(0, 0, 0))
        canvas.paste(panel, (offset, 24))
    return canvas


def save_prediction_outputs(
    raw_path: str | Path,
    image: Image.Image,
    prediction: np.ndarray | Image.Image,
) -> dict[str, Path]:
    """Save class indices separately from human-readable renderings."""
    output = Path(raw_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    array = _mask_array(prediction).astype(np.uint8)
    unexpected = sorted({int(value) for value in np.unique(array)} - {0, 1})
    if unexpected:
        raise ValueError(f"prediction contains unsupported values: {unexpected}")
    raw = output.with_suffix(".png")
    color = raw.with_name(f"{raw.stem}.color.png")
    overlay = raw.with_name(f"{raw.stem}.overlay.png")
    Image.fromarray(array).save(raw)
    colorize_mask(array).save(color)
    prediction_overlay(image, array).save(overlay)
    return {"raw": raw, "color": color, "overlay": overlay}

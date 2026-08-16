"""Schema-driven Pillow mask, overlay, and comparison rendering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..data.schema import ClassDefinition, LabelSchema

BACKGROUND_COLOR = (32, 32, 32)
PET_COLOR = (46, 204, 113)
IGNORE_COLOR = (160, 160, 160)
FALSE_COLOR = (231, 76, 60)
FALSE_NEGATIVE_COLOR = (52, 152, 219)
IGNORED_ERROR_COLOR = (241, 196, 15)


def _binary_schema(ignore_index: int) -> LabelSchema:
    return LabelSchema(
        classes=(
            ClassDefinition(0, "background", BACKGROUND_COLOR),
            ClassDefinition(1, "pet", PET_COLOR),
        ),
        ignore_index=ignore_index,
    )


def _mask_array(mask: np.ndarray | Image.Image) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"mask must be two-dimensional, got {array.shape}")
    return array


def colorize_mask(
    mask: np.ndarray | Image.Image,
    ignore_index: int = 255,
    *,
    schema: LabelSchema | None = None,
) -> Image.Image:
    """Render class indices using the prepared dataset's stable palette."""
    resolved = schema or _binary_schema(ignore_index)
    array = _mask_array(mask)
    allowed = set(range(resolved.num_classes)) | {resolved.ignore_index}
    unexpected = sorted({int(value) for value in np.unique(array)} - allowed)
    if unexpected:
        raise ValueError(f"mask contains unsupported values: {unexpected}")
    colored = np.empty((*array.shape, 3), dtype=np.uint8)
    for item in resolved.classes:
        colored[array == item.id] = item.color
    colored[array == resolved.ignore_index] = IGNORE_COLOR
    return Image.fromarray(colored)


def _blend_colors(image: Image.Image, colors: np.ndarray, selected: np.ndarray, alpha: float) -> Image.Image:
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    original = np.asarray(image.convert("RGB"), dtype=np.float32)
    if selected.shape != original.shape[:2]:
        raise ValueError("image and mask sizes must match")
    result = original.copy()
    result[selected] = (1 - alpha) * original[selected] + alpha * colors[selected]
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def prediction_overlay(
    image: Image.Image,
    prediction: np.ndarray | Image.Image,
    alpha: float = 0.45,
    *,
    schema: LabelSchema | None = None,
) -> Image.Image:
    """Overlay non-background predictions using their schema colors."""
    resolved = schema or _binary_schema(255)
    array = _mask_array(prediction)
    colors = np.asarray(colorize_mask(array, schema=resolved), dtype=np.float32)
    return _blend_colors(image, colors, array != 0, alpha)


def error_overlay(
    image: Image.Image,
    target: np.ndarray | Image.Image,
    prediction: np.ndarray | Image.Image,
    ignore_index: int = 255,
    alpha: float = 0.65,
    *,
    schema: LabelSchema | None = None,
) -> Image.Image:
    """Show correct classes by palette, mistakes in red, and ignored pixels in yellow."""
    resolved = schema or _binary_schema(ignore_index)
    target_array = _mask_array(target)
    prediction_array = _mask_array(prediction)
    if target_array.shape != prediction_array.shape:
        raise ValueError("target and prediction sizes must match")
    if resolved.num_classes == 2 and tuple(item.name for item in resolved.classes) == ("background", "pet"):
        colors = np.zeros((*target_array.shape, 3), dtype=np.float32)
        valid = target_array != resolved.ignore_index
        true_positive = valid & (target_array == 1) & (prediction_array == 1)
        false_positive = valid & (target_array == 0) & (prediction_array == 1)
        false_negative = valid & (target_array == 1) & (prediction_array == 0)
        colors[true_positive] = PET_COLOR
        colors[false_positive] = FALSE_COLOR
        colors[false_negative] = FALSE_NEGATIVE_COLOR
        colors[~valid] = IGNORED_ERROR_COLOR
        selected = true_positive | false_positive | false_negative | ~valid
        return _blend_colors(image, colors, selected, alpha)
    colors = np.asarray(colorize_mask(prediction_array, schema=resolved), dtype=np.float32)
    valid = target_array != resolved.ignore_index
    colors[valid & (target_array != prediction_array)] = FALSE_COLOR
    colors[~valid] = IGNORED_ERROR_COLOR
    return _blend_colors(image, colors, np.ones(valid.shape, dtype=bool), alpha)


def comparison_grid(
    image: Image.Image,
    target: np.ndarray | Image.Image,
    prediction: np.ndarray | Image.Image,
    ignore_index: int = 255,
    *,
    schema: LabelSchema | None = None,
) -> Image.Image:
    """Create original, target, prediction, and error panels with labels."""
    resolved = schema or _binary_schema(ignore_index)
    original = image.convert("RGB")
    panels = (
        original,
        colorize_mask(target, schema=resolved),
        colorize_mask(prediction, schema=resolved),
        error_overlay(original, target, prediction, schema=resolved),
    )
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
    *,
    schema: LabelSchema | None = None,
) -> dict[str, Path]:
    """Save class indices separately from palette-driven renderings."""
    resolved = schema or _binary_schema(255)
    output = Path(raw_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    array = _mask_array(prediction)
    unexpected = sorted({int(value) for value in np.unique(array)} - set(range(resolved.num_classes)))
    if unexpected:
        raise ValueError(f"prediction contains unsupported values: {unexpected}")
    raw_array = array.astype(np.uint8 if resolved.num_classes <= 256 else np.uint16)
    raw = output.with_suffix(".png")
    color = raw.with_name(f"{raw.stem}.color.png")
    overlay = raw.with_name(f"{raw.stem}.overlay.png")
    Image.fromarray(raw_array).save(raw)
    colorize_mask(array, schema=resolved).save(color)
    prediction_overlay(image, array, schema=resolved).save(overlay)
    return {"raw": raw, "color": color, "overlay": overlay}

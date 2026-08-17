"""Prepared-dataset statistics and schema-driven visual previews."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .masks import MaskDecoder
from .metadata import PreparedDatasetMetadata, read_dataset_metadata
from .schema import ClassDefinition, LabelSchema

HEADER_HEIGHT = 24


def _resolve(manifest: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (manifest.parent / path).resolve()


def _legacy_metadata(manifest: Path) -> PreparedDatasetMetadata:
    schema = LabelSchema(
        classes=(
            ClassDefinition(0, "background", (32, 32, 32)),
            ClassDefinition(1, "pet", (46, 204, 113)),
        ),
        ignore_index=255,
    )
    return PreparedDatasetMetadata(
        provider="legacy_oxford",
        split_method="unknown",
        seed=None,
        data_dir=".",
        manifests={"preview": manifest.name},
        split_counts={"preview": 0},
        manifest_sha256={"preview": "not-validated"},
        label_schema=schema,
        label_schema_sha256="not-validated",
        label_map={1: 1, 2: 0},
        ignore_values=(3,),
    )


def _metadata_for_manifest(manifest: Path) -> PreparedDatasetMetadata:
    path = manifest.parent / "dataset.yaml"
    return read_dataset_metadata(path) if path.is_file() else _legacy_metadata(manifest)


def _raw_panel(raw: np.ndarray) -> Image.Image:
    if raw.ndim == 3 and raw.shape[2] == 3:
        return Image.fromarray(raw.astype(np.uint8))
    values = sorted(int(value) for value in np.unique(raw))
    rendered = np.zeros(raw.shape, dtype=np.uint8)
    denominator = max(1, len(values) - 1)
    for index, value in enumerate(values):
        rendered[raw == value] = round(255 * index / denominator)
    return Image.fromarray(rendered).convert("RGB")


def create_dataset_preview(
    manifest: str | Path,
    output: str | Path,
    *,
    limit: int = 4,
) -> Path:
    """Render image, raw mask, mapped mask, and overlay panels."""
    from ..evaluation.visualization import colorize_mask, prediction_overlay

    if limit <= 0:
        raise ValueError("limit must be positive")
    manifest_path = Path(manifest).resolve()
    with manifest_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))[:limit]
    if not rows:
        raise ValueError(f"manifest contains no samples: {manifest_path}")
    required = {"sample_id", "image_path", "mask_path"}
    if not required <= set(rows[0]):
        raise ValueError(f"manifest is missing columns: {sorted(required - set(rows[0]))}")
    metadata = _metadata_for_manifest(manifest_path)
    schema = metadata.label_schema
    decoder = MaskDecoder(
        schema,
        indexed_map=metadata.label_map,
        ignore_values=metadata.ignore_values,
    )
    rendered_rows: list[Image.Image] = []
    panel_size: tuple[int, int] | None = None
    for row in rows:
        with Image.open(_resolve(manifest_path, row["image_path"])) as opened:
            image = opened.convert("RGB")
        with Image.open(_resolve(manifest_path, row["mask_path"])) as opened:
            raw = np.asarray(opened).copy()
            mapped = decoder(opened).numpy()
        if panel_size is None:
            panel_size = image.size
        image = image.resize(panel_size, Image.Resampling.BILINEAR)
        raw_image = _raw_panel(raw).resize(panel_size, Image.Resampling.NEAREST)
        mapped_image = Image.fromarray(mapped.astype(np.int32)).resize(panel_size, Image.Resampling.NEAREST)
        mapped = np.asarray(mapped_image)
        panels = (
            image,
            raw_image,
            colorize_mask(mapped, schema=schema),
            prediction_overlay(image, mapped, schema=schema),
        )
        labels = ("Image", "Raw mask", "Mapped mask", "Overlay")
        row_image = Image.new("RGB", (panel_size[0] * 4, panel_size[1] + HEADER_HEIGHT), color="white")
        draw = ImageDraw.Draw(row_image)
        for index, (panel, label) in enumerate(zip(panels, labels, strict=True)):
            offset = index * panel_size[0]
            draw.text((offset + 4, 5), label, fill="black")
            row_image.paste(panel, (offset, HEADER_HEIGHT))
        rendered_rows.append(row_image)
    canvas = Image.new("RGB", (rendered_rows[0].width, sum(row.height for row in rendered_rows)), color="white")
    top = 0
    for rendered_row in rendered_rows:
        canvas.paste(rendered_row, (0, top))
        top += rendered_row.height
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)
    return destination


def inspect_prepared_data(
    manifest_dir: str | Path,
    *,
    split: str = "train",
    limit: int = 16,
    preview: str | Path | None = None,
) -> dict[str, Any]:
    """Inspect declared and mapped labels from a bounded prepared-data sample."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    root = Path(manifest_dir).resolve()
    metadata = read_dataset_metadata(root / "dataset.yaml")
    if split not in metadata.manifests:
        raise ValueError(f"unknown prepared split: {split}")
    manifest = root / metadata.manifests[split]
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))[:limit]
    decoder = MaskDecoder(
        metadata.label_schema,
        indexed_map=metadata.label_map,
        ignore_values=metadata.ignore_values,
    )
    raw_values: set[int] = set()
    raw_colors: set[tuple[int, int, int]] = set()
    mapped_counts = {item.id: 0 for item in metadata.label_schema.classes}
    ignored_pixels = 0
    total_pixels = 0
    for row in rows:
        with Image.open(_resolve(manifest, row["mask_path"])) as opened:
            raw = np.asarray(opened)
            mapped = decoder(opened).numpy()
        if raw.ndim == 2:
            raw_values.update(int(value) for value in np.unique(raw))
        else:
            raw_colors.update(
                (int(color[0]), int(color[1]), int(color[2])) for color in np.unique(raw.reshape(-1, 3), axis=0)
            )
        for class_id in mapped_counts:
            mapped_counts[class_id] += int((mapped == class_id).sum())
        ignored_pixels += int((mapped == metadata.label_schema.ignore_index).sum())
        total_pixels += mapped.size
    valid_pixels = total_pixels - ignored_pixels
    report: dict[str, Any] = {
        "provider": metadata.provider,
        "split": split,
        "split_counts": metadata.split_counts,
        "inspected_samples": len(rows),
        "raw_values": sorted(raw_values),
        "mapped_values": sorted(
            [class_id for class_id, count in mapped_counts.items() if count]
            + ([metadata.label_schema.ignore_index] if ignored_pixels else [])
        ),
        "class_ratios": {
            item.name: mapped_counts[item.id] / valid_pixels if valid_pixels else 0.0
            for item in metadata.label_schema.classes
        },
        "ignore_ratio": ignored_pixels / total_pixels if total_pixels else 0.0,
        "classes": [
            {"id": item.id, "name": item.name, "color": list(item.color)} for item in metadata.label_schema.classes
        ],
    }
    if raw_colors:
        report["raw_colors"] = [list(color) for color in sorted(raw_colors)]
    if preview is not None:
        report["preview"] = str(create_dataset_preview(manifest, preview, limit=min(limit, len(rows))))
    return report

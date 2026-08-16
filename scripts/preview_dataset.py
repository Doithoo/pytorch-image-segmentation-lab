"""Create a visual preview from a segmentation manifest."""

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from image_segmenter.evaluation.visualization import colorize_mask, prediction_overlay

HEADER_HEIGHT = 24
OXFORD_LABEL_MAP = {1: 1, 2: 0}
OXFORD_IGNORE_VALUES = {3}


def _resolve(manifest: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (manifest.parent / path).resolve()


def _map_oxford_mask(raw: np.ndarray) -> np.ndarray:
    unexpected = sorted(
        {int(value) for value in np.unique(raw)}
        - set(OXFORD_LABEL_MAP)
        - OXFORD_IGNORE_VALUES
    )
    if unexpected:
        raise ValueError(f"mask contains unsupported Oxford labels: {unexpected}")
    mapped = np.full(raw.shape, 255, dtype=np.uint8)
    for raw_value, class_index in OXFORD_LABEL_MAP.items():
        mapped[raw == raw_value] = class_index
    return mapped


def _raw_mask_panel(raw: np.ndarray) -> Image.Image:
    scale = {1: 85, 2: 170, 3: 255}
    rendered = np.zeros(raw.shape, dtype=np.uint8)
    for value, intensity in scale.items():
        rendered[raw == value] = intensity
    return Image.fromarray(rendered).convert("RGB")


def create_dataset_preview(
    manifest: str | Path,
    output: str | Path,
    *,
    limit: int = 4,
) -> Path:
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

    rendered_rows: list[Image.Image] = []
    panel_size: tuple[int, int] | None = None
    for row in rows:
        with Image.open(_resolve(manifest_path, row["image_path"])) as source:
            image = source.convert("RGB")
        with Image.open(_resolve(manifest_path, row["mask_path"])) as source:
            raw = np.asarray(source.convert("L"))
        if panel_size is None:
            panel_size = image.size
        image = image.resize(panel_size, Image.Resampling.BILINEAR)
        raw_image = Image.fromarray(raw).resize(panel_size, Image.Resampling.NEAREST)
        raw = np.asarray(raw_image)
        mapped = _map_oxford_mask(raw)
        foreground = (mapped == 1).astype(np.uint8)
        panels = (
            image,
            _raw_mask_panel(raw),
            colorize_mask(mapped),
            prediction_overlay(image, foreground),
        )
        labels = ("Image", "Raw mask", "Mapped mask", "Overlay")
        row_image = Image.new(
            "RGB",
            (panel_size[0] * len(panels), panel_size[1] + HEADER_HEIGHT),
            color="white",
        )
        draw = ImageDraw.Draw(row_image)
        for index, (panel, label) in enumerate(zip(panels, labels, strict=True)):
            offset = index * panel_size[0]
            draw.text((offset + 4, 5), label, fill="black")
            row_image.paste(panel, (offset, HEADER_HEIGHT))
        rendered_rows.append(row_image)

    canvas = Image.new(
        "RGB",
        (rendered_rows[0].width, sum(row.height for row in rendered_rows)),
        color="white",
    )
    top = 0
    for row in rendered_rows:
        canvas.paste(row, (0, top))
        top += row.height
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)
    return destination


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preview samples from a segmentation manifest.")
    parser.add_argument("manifest", type=Path, help="CSV manifest to preview")
    parser.add_argument("--output", type=Path, default=Path("preview.png"))
    parser.add_argument("--limit", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output = create_dataset_preview(args.manifest, args.output, limit=args.limit)
    print(f"Saved dataset preview to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

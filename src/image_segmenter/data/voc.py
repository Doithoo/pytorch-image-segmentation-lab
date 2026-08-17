"""Pascal VOC 2012 semantic-segmentation preparation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .manifest import ManifestError, ManifestRow, _relative_path
from .providers import _write_rows
from .schema import ClassDefinition, LabelSchema

VOC_CLASS_NAMES = (
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)

VOC_COLORS = (
    (0, 0, 0),
    (128, 0, 0),
    (0, 128, 0),
    (128, 128, 0),
    (0, 0, 128),
    (128, 0, 128),
    (0, 128, 128),
    (128, 128, 128),
    (64, 0, 0),
    (192, 0, 0),
    (64, 128, 0),
    (192, 128, 0),
    (64, 0, 128),
    (192, 0, 128),
    (64, 128, 128),
    (192, 128, 128),
    (0, 64, 0),
    (128, 64, 0),
    (0, 192, 0),
    (128, 192, 0),
    (0, 64, 128),
)

VOC_SCHEMA = LabelSchema(
    classes=tuple(
        ClassDefinition(class_id, name, color)
        for class_id, (name, color) in enumerate(zip(VOC_CLASS_NAMES, VOC_COLORS, strict=True))
    ),
    ignore_index=255,
)
VOC_GUIDE = "docs/reference/voc2012.md"


def _voc_root(data_dir: Path) -> Path:
    candidates = (data_dir, data_dir / "VOCdevkit" / "VOC2012", data_dir / "VOC2012")
    for candidate in candidates:
        if (candidate / "JPEGImages").is_dir() and (candidate / "SegmentationClass").is_dir():
            return candidate
    raise FileNotFoundError(
        f"VOC 2012 directory not found under {data_dir}; see {VOC_GUIDE} for the expected local layout"
    )


def _read_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"VOC split list not found: {path}; see {VOC_GUIDE}")
    values = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not values or len(values) != len(set(values)):
        raise ManifestError(f"VOC split list must contain unique sample IDs: {path}")
    return values


def prepare_voc2012_provider(**kwargs: Any) -> dict[str, Path]:
    params = dict(kwargs.pop("params", {}))
    unknown = sorted(set(params) - {"test_policy", "test_list"})
    if unknown:
        raise ValueError(f"unknown dataset provider parameter: {unknown[0]}")
    kwargs.pop("split_ratios")
    source = _voc_root(Path(kwargs["data_dir"]).resolve())
    output = Path(kwargs["manifest_dir"]).resolve()
    split_root = source / "ImageSets" / "Segmentation"
    memberships = {
        "train": _read_ids(split_root / "train.txt"),
        "valid": _read_ids(split_root / "val.txt"),
    }
    if "test_list" in params:
        memberships["test"] = _read_ids(Path(str(params["test_list"])).resolve())
    elif params.get("test_policy") == "use_valid":
        memberships["test"] = list(memberships["valid"])
    else:
        raise ValueError("VOC requires data.params.test_policy: use_valid or an explicit data.params.test_list")

    rows_by_split: dict[str, list[dict[str, object]]] = {}
    allowed = set(range(21)) | {255}
    for split, sample_ids in memberships.items():
        rows: list[dict[str, object]] = []
        for sample_id in sample_ids:
            image_path = source / "JPEGImages" / f"{sample_id}.jpg"
            mask_path = source / "SegmentationClass" / f"{sample_id}.png"
            if not image_path.is_file() or not mask_path.is_file():
                raise FileNotFoundError(f"VOC labeled sample files not found for {sample_id}; see {VOC_GUIDE}")
            with Image.open(image_path) as image:
                width, height = image.size
            with Image.open(mask_path) as mask:
                mask.load()
                if mask.size != (width, height):
                    raise ManifestError(f"VOC image and mask size mismatch for {sample_id}")
                values = {int(value) for value in np.unique(np.asarray(mask))}
            unknown_values = sorted(values - allowed)
            if unknown_values:
                raise ManifestError(f"VOC mask {sample_id} contains unknown values: {unknown_values}")
            rows.append(
                vars(
                    ManifestRow(
                        sample_id=sample_id,
                        image_path=_relative_path(image_path, output),
                        mask_path=_relative_path(mask_path, output),
                        split=split,
                        width=width,
                        height=height,
                        source_split="val" if split in {"valid", "test"} else "train",
                    )
                )
            )
        rows_by_split[split] = rows
    return _write_rows(
        output=output,
        source=source,
        rows_by_split=rows_by_split,
        provider="voc2012",
        split_method="official_train_val",
        seed=None,
        label_map={class_id: class_id for class_id in range(21)},
        ignore_values={255},
        label_schema=VOC_SCHEMA,
        audit={"split_reuse": "valid"} if params.get("test_policy") == "use_valid" else None,
    )

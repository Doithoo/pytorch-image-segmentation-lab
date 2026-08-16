"""Checkpoint evaluation with numeric tables and bounded visual evidence."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from ..config import ExperimentConfig, config_from_dict
from ..data import SegmentationDataset, build_transform
from ..models import create_model
from ..training.checkpoint import load_checkpoint
from ..training.train import resolve_device
from .metrics import SegmentationMetrics, per_image_scores, serializable_metrics
from .visualization import comparison_grid


def _sample_limit(config: ExperimentConfig, split: str) -> int | None:
    return {
        "train": config.data.max_train_samples,
        "valid": config.data.max_valid_samples,
        "test": config.data.max_test_samples,
    }[split]


def _denormalize(image: torch.Tensor, config: ExperimentConfig) -> Image.Image:
    mean = torch.tensor(config.data.normalize_mean, dtype=image.dtype).view(3, 1, 1)
    std = torch.tensor(config.data.normalize_std, dtype=image.dtype).view(3, 1, 1)
    array = ((image.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(array)


def _finite_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else math.nan


def _per_class_rows(metrics: SegmentationMetrics) -> list[dict[str, Any]]:
    counts = metrics.confusion.to(torch.float64)
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(("background", "pet")):
        intersection = float(counts[index, index].item())
        target_count = float(counts[index].sum().item())
        predicted_count = float(counts[:, index].sum().item())
        union = target_count + predicted_count - intersection
        dice_denominator = target_count + predicted_count
        rows.append(
            {
                "class_index": index,
                "class_name": name,
                "iou": intersection / union if union else None,
                "dice": 2 * intersection / dice_denominator if dice_denominator else None,
                "target_pixels": int(target_count),
                "predicted_pixels": int(predicted_count),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: "" if isinstance(value, float) and math.isnan(value) else value for key, value in row.items()}
            )


def _render_samples(
    output_dir: Path,
    folder: str,
    sample_ids: list[str],
    dataset: SegmentationDataset,
    model: torch.nn.Module,
    device: torch.device,
    config: ExperimentConfig,
) -> None:
    destination = output_dir / folder
    destination.mkdir(parents=True, exist_ok=True)
    indices = {row["sample_id"]: index for index, row in enumerate(dataset.rows)}
    for sample_id in sample_ids:
        image, target, _ = dataset[indices[sample_id]]
        with torch.no_grad():
            prediction = model(image.unsqueeze(0).to(device)).argmax(dim=1).squeeze(0).cpu()
        grid = comparison_grid(
            _denormalize(image, config),
            target.numpy().astype(np.uint8),
            prediction.numpy().astype(np.uint8),
            config.loss.ignore_index,
        )
        safe_name = sample_id.replace("/", "__")
        grid.save(destination / f"{safe_name}.png")


def plot_loss_curve(metrics_csv: str | Path, output: str | Path) -> Path:
    """Plot train and validation loss from a run metrics table."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError('plotting requires: uv pip install -e ".[plot]"') from exc
    with Path(metrics_csv).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("training metrics contain no epochs")
    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]
    valid_loss = [float(row["valid_loss"]) for row in rows]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(epochs, train_loss, marker="o", label="train loss")
    axis.plot(epochs, valid_loss, marker="o", label="valid loss")
    axis.set_xlabel("epoch")
    axis.set_ylabel("loss")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    destination = Path(output)
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination


def evaluate_checkpoint(
    checkpoint: str | Path,
    *,
    split: str = "test",
    device: str = "auto",
    output_dir: str | Path | None = None,
    sample_limit: int = 8,
    error_limit: int = 12,
    plot: bool = False,
) -> Path:
    """Evaluate one checkpoint and return its evaluation directory."""
    if split not in {"train", "valid", "test"}:
        raise ValueError("split must be train, valid, or test")
    if sample_limit < 0 or error_limit < 0:
        raise ValueError("visual sample limits must be non-negative")
    checkpoint_record = Path(checkpoint).as_posix()
    checkpoint_path = Path(checkpoint).resolve()
    saved = load_checkpoint(checkpoint_path)
    config = config_from_dict(saved["config"])
    runtime_device = resolve_device(device)
    model = create_model(
        config.model.name,
        in_channels=config.model.in_channels,
        num_classes=config.model.num_classes,
        base_channels=config.model.base_channels,
        pretrained=False,
    )
    model.load_state_dict(saved["model_state"])
    model.to(runtime_device).eval()
    manifest = Path(config.data.manifest_dir) / f"{split}.csv"
    dataset = SegmentationDataset(
        manifest,
        build_transform(config.data, training=False),
        config.data.label_map,
        config.data.ignore_values,
        config.loss.ignore_index,
        _sample_limit(config, split),
    )
    loader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
    aggregate = SegmentationMetrics(config.model.num_classes, config.loss.ignore_index)
    per_image_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for images, targets, sample_ids in loader:
            logits = model(images.to(runtime_device))
            predictions = logits.argmax(dim=1).cpu()
            aggregate.update(predictions, targets)
            for sample_id, scores in zip(
                sample_ids, per_image_scores(predictions, targets, config.loss.ignore_index), strict=True
            ):
                per_image_rows.append({"sample_id": sample_id, **scores})

    global_metrics = aggregate.compute()
    pet_ious = [float(row["pet_iou"]) for row in per_image_rows]
    pet_dice = [float(row["pet_dice"]) for row in per_image_rows]
    summary = {
        "checkpoint": checkpoint_record,
        "split": split,
        "sample_count": len(per_image_rows),
        "global": global_metrics,
        "per_image_mean": {"pet_iou": _finite_mean(pet_ious), "pet_dice": _finite_mean(pet_dice)},
        "empty_target_count": sum(float(row["target_foreground_ratio"]) == 0 for row in per_image_rows),
    }
    destination = Path(output_dir) if output_dir is not None else checkpoint_path.parent / "evaluation"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "metrics.json").write_text(
        json.dumps(serializable_metrics(summary), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    class_rows = _per_class_rows(aggregate)
    _write_csv(destination / "per_class_metrics.csv", class_rows, list(class_rows[0]))
    per_image_rows.sort(key=lambda row: str(row["sample_id"]))
    image_fields = [
        "sample_id",
        "pet_iou",
        "pet_dice",
        "target_foreground_ratio",
        "predicted_foreground_ratio",
        "valid_pixels",
        "ignored_pixels",
    ]
    _write_csv(destination / "per_image_metrics.csv", per_image_rows, image_fields)

    sample_ids = [str(row["sample_id"]) for row in per_image_rows[:sample_limit]]
    worst_rows = sorted(
        (row for row in per_image_rows if math.isfinite(float(row["pet_iou"]))),
        key=lambda row: float(row["pet_iou"]),
    )
    worst_ids = [str(row["sample_id"]) for row in worst_rows[:error_limit]]
    _render_samples(destination, "sample_overlays", sample_ids, dataset, model, runtime_device, config)
    _render_samples(destination, "worst_cases", worst_ids, dataset, model, runtime_device, config)
    if plot:
        plot_loss_curve(checkpoint_path.parent / "metrics.csv", destination / "loss_curve.png")
    return destination

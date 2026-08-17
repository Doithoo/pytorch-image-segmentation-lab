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
from ..data import LabelSchema, SegmentationDataset, build_transform, load_external_datasets, read_dataset_metadata
from ..models import create_model
from ..training.checkpoint import checkpoint_label_schema, load_checkpoint
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


def _per_class_rows(metrics: SegmentationMetrics, schema: LabelSchema) -> list[dict[str, Any]]:
    counts = metrics.confusion.to(torch.float64)
    rows: list[dict[str, Any]] = []
    for index, definition in enumerate(schema.classes):
        intersection = float(counts[index, index].item())
        target_count = float(counts[index].sum().item())
        predicted_count = float(counts[:, index].sum().item())
        union = target_count + predicted_count - intersection
        dice_denominator = target_count + predicted_count
        rows.append(
            {
                "class_index": index,
                "class_name": definition.name,
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


def _clear_managed_visuals(output_dir: Path) -> None:
    for folder in ("sample_overlays", "worst_cases"):
        for image in (output_dir / folder).glob("*.png"):
            image.unlink()
    (output_dir / "loss_curve.png").unlink(missing_ok=True)


def _render_samples(
    output_dir: Path,
    folder: str,
    sample_ids: list[str],
    dataset: SegmentationDataset,
    model: torch.nn.Module,
    device: torch.device,
    config: ExperimentConfig,
    label_schema: LabelSchema,
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
            target.numpy(),
            prediction.numpy(),
            label_schema.ignore_index,
            schema=label_schema,
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
    label_schema = checkpoint_label_schema(saved)
    runtime_device = resolve_device(device)
    model = create_model(
        config.model.name,
        factory=config.model.factory,
        in_channels=config.model.in_channels,
        num_classes=label_schema.num_classes,
        pretrained=False,
        params=config.model.params,
    )
    model.load_state_dict(saved["model_state"])
    model.to(runtime_device).eval()
    metadata = read_dataset_metadata(Path(config.data.manifest_dir) / "dataset.yaml")
    if metadata.label_schema != label_schema:
        raise ValueError("checkpoint label schema does not match prepared dataset metadata")
    manifest = Path(config.data.manifest_dir) / metadata.manifests[split]
    if config.data.factory is not None:
        datasets = load_external_datasets(
            config.data.factory,
            data_dir=config.data.data_dir,
            manifest_dir=Path(config.data.manifest_dir),
            transforms={
                item: build_transform(config.data, training=False, ignore_index=config.loss.ignore_index)
                for item in ("train", "valid", "test")
            },
            label_schema=label_schema,
            params=config.data.params,
        )
        dataset = datasets[split]
    else:
        dataset = SegmentationDataset(
            manifest,
            build_transform(config.data, training=False, ignore_index=config.loss.ignore_index),
            metadata.label_map,
            list(metadata.ignore_values),
            label_schema.ignore_index,
            _sample_limit(config, split),
            label_schema=label_schema,
        )
    loader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
    class_names = tuple(item.name for item in label_schema.classes)
    aggregate = SegmentationMetrics(
        label_schema.num_classes,
        label_schema.ignore_index,
        class_names=class_names,
    )
    per_image_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for images, targets, sample_ids in loader:
            logits = model(images.to(runtime_device))
            predictions = logits.argmax(dim=1).cpu()
            aggregate.update(predictions, targets)
            for sample_id, scores in zip(
                sample_ids,
                per_image_scores(
                    predictions,
                    targets,
                    label_schema.ignore_index,
                    label_schema.num_classes,
                    class_names,
                ),
                strict=True,
            ):
                per_image_rows.append({"sample_id": sample_id, **scores})

    global_metrics = aggregate.compute()
    mean_ious = [float(row["mean_iou"]) for row in per_image_rows]
    mean_dices = [float(row["mean_dice"]) for row in per_image_rows]
    summary = {
        "checkpoint": checkpoint_record,
        "split": split,
        "sample_count": len(per_image_rows),
        "global": global_metrics,
        "per_image_mean": {"mean_iou": _finite_mean(mean_ious), "mean_dice": _finite_mean(mean_dices)},
    }
    destination = Path(output_dir) if output_dir is not None else checkpoint_path.parent / "evaluation"
    destination.mkdir(parents=True, exist_ok=True)
    _clear_managed_visuals(destination)
    (destination / "metrics.json").write_text(
        json.dumps(serializable_metrics(summary), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    class_rows = _per_class_rows(aggregate, label_schema)
    _write_csv(destination / "per_class_metrics.csv", class_rows, list(class_rows[0]))
    per_image_rows.sort(key=lambda row: str(row["sample_id"]))
    image_fields = ["sample_id", *[key for key in per_image_rows[0] if key != "sample_id"]]
    _write_csv(destination / "per_image_metrics.csv", per_image_rows, image_fields)

    sample_ids = [str(row["sample_id"]) for row in per_image_rows[:sample_limit]]
    worst_rows = sorted(
        (row for row in per_image_rows if math.isfinite(float(row["mean_iou"]))),
        key=lambda row: float(row["mean_iou"]),
    )
    worst_ids = [str(row["sample_id"]) for row in worst_rows[:error_limit]]
    if isinstance(dataset, SegmentationDataset):
        _render_samples(
            destination, "sample_overlays", sample_ids, dataset, model, runtime_device, config, label_schema
        )
        _render_samples(destination, "worst_cases", worst_ids, dataset, model, runtime_device, config, label_schema)
    if plot:
        plot_loss_curve(checkpoint_path.parent / "metrics.csv", destination / "loss_curve.png")
    return destination

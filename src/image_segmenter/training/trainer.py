"""Readable epoch loop and checkpoint lifecycle."""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from ..config import ExperimentConfig
from ..data import LabelSchema
from ..evaluation.metrics import SegmentationMetrics
from .checkpoint import save_checkpoint
from .losses import SegmentationLoss
from .precision import PrecisionController

METRIC_FIELDS = ("epoch", "train_loss", "train_pet_iou", "valid_loss", "valid_pet_iou", "lr")


class Trainer:
    """Train one segmentation model and persist observable run state."""

    def __init__(
        self,
        model: nn.Module,
        criterion: SegmentationLoss,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        device: torch.device,
        config: ExperimentConfig,
        run_dir: Path,
        manifest_checksums: dict[str, str],
        label_schema: LabelSchema,
        precision: PrecisionController,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.run_dir = run_dir
        self.manifest_checksums = manifest_checksums
        self.label_schema = label_schema
        self.precision = precision

    def _metric_keys(self) -> list[str]:
        keys = ["mean_iou", "mean_dice", "pixel_accuracy"]
        for item in self.label_schema.classes:
            keys.extend((f"class_iou:{item.name}", f"class_dice:{item.name}"))
        if tuple(item.name for item in self.label_schema.classes) == ("background", "pet"):
            keys.extend(("background_iou", "pet_iou", "pet_dice", "macro_dice"))
        return keys

    def _metric_fields(self) -> list[str]:
        keys = self._metric_keys()
        return [
            "epoch",
            "train_loss",
            *(f"train_{key}" for key in keys),
            "valid_loss",
            *(f"valid_{key}" for key in keys),
            "lr",
            "selected_metric",
            "epoch_duration_seconds",
        ]

    def _record_fit_runtime(self, elapsed_seconds: float, start_epoch: int) -> None:
        path = self.run_dir / "run.yaml"
        metadata = yaml.safe_load(path.read_text())
        if not isinstance(metadata, dict):
            raise ValueError(f"run metadata is invalid: {path}")
        fit_history = metadata.setdefault("fit_history", [])
        if not isinstance(fit_history, list):
            raise ValueError(f"run metadata fit_history is invalid: {path}")
        fit_history.append(
            {
                "start_epoch": start_epoch,
                "completed_epoch": self.config.train.epochs,
                "device": str(self.device),
                "duration_seconds": elapsed_seconds,
            }
        )
        metadata["last_fit_seconds"] = elapsed_seconds
        metadata["total_training_seconds"] = float(metadata.get("total_training_seconds", 0.0)) + elapsed_seconds
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(yaml.safe_dump(metadata, sort_keys=False))
        temporary.replace(path)

    def _reconcile_resume_metrics(self, metrics_path: Path, completed_epoch: int) -> None:
        if not metrics_path.is_file():
            raise ValueError(f"resume metrics file not found: {metrics_path}")
        with metrics_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fieldnames = reader.fieldnames
        expected_fields = self._metric_fields()
        if fieldnames != expected_fields:
            raise ValueError(f"resume metrics columns do not match the current run: {metrics_path}")
        if len(rows) < completed_epoch:
            raise ValueError(f"resume metrics end before last.pt epoch {completed_epoch}: {metrics_path}")
        try:
            recorded_epochs = [int(row["epoch"]) for row in rows[:completed_epoch]]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"resume metrics contain an invalid epoch: {metrics_path}") from exc
        expected_epochs = list(range(1, completed_epoch + 1))
        if recorded_epochs != expected_epochs:
            raise ValueError(
                f"resume metrics epochs do not match last.pt through epoch {completed_epoch}: {metrics_path}"
            )
        if len(rows) == completed_epoch:
            return
        temporary = metrics_path.with_suffix(metrics_path.suffix + ".tmp")
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=expected_fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows[:completed_epoch])
        temporary.replace(metrics_path)

    def dry_run(self, loader: DataLoader[Any]) -> None:
        """Execute one real optimization update without creating run artifacts."""
        try:
            images, masks, _sample_ids = next(iter(loader))
        except StopIteration as exc:
            raise ValueError("training loader contains no batches") from exc
        images = images.to(self.device)
        masks = masks.to(self.device)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        with self.precision.autocast():
            logits = self.model(images)
            loss, _components = self.criterion(logits, masks)
        self.precision.backward_step(
            loss,
            self.optimizer,
            self.model.parameters(),
            grad_clip=self.config.train.grad_clip,
        )
        valid_ratio = float((masks != self.config.loss.ignore_index).float().mean().item())
        print(f"device: {self.device}")
        print(f"images: {tuple(images.shape)}")
        print(f"masks: {tuple(masks.shape)} values={sorted(masks.unique().tolist())}")
        print(f"logits: {tuple(logits.shape)}")
        print(f"valid pixel ratio: {valid_ratio:.4f}")
        print(f"loss: {loss.detach().item():.6f}")
        print("dry-run OK")

    def _run_epoch(self, loader: DataLoader[Any], *, training: bool) -> dict[str, float | int]:
        if training:
            self.model.train()
        else:
            self.model.eval()
        metrics = SegmentationMetrics(
            num_classes=self.label_schema.num_classes,
            ignore_index=self.label_schema.ignore_index,
            class_names=tuple(item.name for item in self.label_schema.classes),
        )
        loss_sum = 0.0
        sample_count = 0
        for images, masks, _sample_ids in loader:
            images = images.to(self.device)
            masks = masks.to(self.device)
            if training:
                self.optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training), self.precision.autocast():
                logits = self.model(images)
                loss, _components = self.criterion(logits, masks)
            if training:
                self.precision.backward_step(
                    loss,
                    self.optimizer,
                    self.model.parameters(),
                    grad_clip=self.config.train.grad_clip,
                )
            metrics.update(logits.detach(), masks)
            loss_sum += loss.detach().item() * images.shape[0]
            sample_count += images.shape[0]
        if sample_count == 0:
            raise ValueError("data loader contains no samples")
        result = metrics.compute()
        result["loss"] = loss_sum / sample_count
        return result

    def fit(
        self,
        train_loader: DataLoader[Any],
        valid_loader: DataLoader[Any],
        *,
        start_epoch: int = 1,
        best_metric: float = -math.inf,
    ) -> Path:
        """Train through the configured final epoch and write run artifacts."""
        fit_started = time.monotonic()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = self.run_dir / "metrics.csv"
        append = start_epoch > 1
        if append:
            self._reconcile_resume_metrics(metrics_path, start_epoch - 1)
        with metrics_path.open("a" if append else "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._metric_fields(), lineterminator="\n")
            if not append:
                writer.writeheader()
            for epoch in range(start_epoch, self.config.train.epochs + 1):
                epoch_started = time.monotonic()
                learning_rate = float(self.optimizer.param_groups[0]["lr"])
                train_metrics = self._run_epoch(train_loader, training=True)
                with torch.no_grad():
                    valid_metrics = self._run_epoch(valid_loader, training=False)
                if self.scheduler is not None:
                    self.scheduler.step()
                row: dict[str, object] = {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "valid_loss": valid_metrics["loss"],
                    "lr": learning_rate,
                    "selected_metric": self.config.train.best_metric,
                }
                for key in self._metric_keys():
                    row[f"train_{key}"] = train_metrics[key]
                    row[f"valid_{key}"] = valid_metrics[key]
                row["epoch_duration_seconds"] = time.monotonic() - epoch_started
                writer.writerow(row)
                handle.flush()
                metric_name = self.config.train.best_metric
                if metric_name not in valid_metrics:
                    raise ValueError(f"train.best_metric is not available for this schema: {metric_name}")
                valid_score = float(valid_metrics[metric_name])
                improved = math.isfinite(valid_score) and (not math.isfinite(best_metric) or valid_score > best_metric)
                if improved or not (self.run_dir / "best.pt").exists():
                    best_metric = valid_score
                    save_checkpoint(
                        self.run_dir / "best.pt",
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        epoch=epoch,
                        best_metric=best_metric,
                        cfg=self.config,
                        manifest_checksums=self.manifest_checksums,
                        label_schema=self.label_schema,
                        precision_state=self.precision.state_dict(),
                    )
                save_checkpoint(
                    self.run_dir / "last.pt",
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch=epoch,
                    best_metric=best_metric,
                    cfg=self.config,
                    manifest_checksums=self.manifest_checksums,
                    label_schema=self.label_schema,
                    precision_state=self.precision.state_dict(),
                )
                print(
                    f"epoch {epoch}/{self.config.train.epochs} "
                    f"train_loss={float(train_metrics['loss']):.4f} "
                    f"valid_{metric_name}={valid_score:.4f}"
                )
        self._record_fit_runtime(time.monotonic() - fit_started, start_epoch)
        return self.run_dir

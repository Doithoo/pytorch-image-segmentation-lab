"""Assemble configured data, model, loss, optimizer, and Trainer."""

from __future__ import annotations

import hashlib
import platform
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torchvision
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.utils.data import DataLoader

from .. import __version__
from ..config import ExperimentConfig, config_from_dict, dump_config, to_dict
from ..data import SegmentationDataset, build_transform
from ..models import create_model
from .checkpoint import git_revision, load_checkpoint, restore_training_state
from .losses import SegmentationLoss
from .trainer import Trainer


def resolve_device(requested: str) -> torch.device:
    """Resolve an explicit device or choose the best available accelerator."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was explicitly requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was explicitly requested but is unavailable")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    """Seed process-level random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int) -> None:
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_run_metadata(
    path: Path,
    device: torch.device,
    manifest_checksums: dict[str, str],
) -> None:
    if path.exists():
        return
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "package_version": __version__,
        "git_revision": git_revision(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "pytorch": str(torch.__version__),
        "torchvision": torchvision.__version__,
        "device": str(device),
        "accelerators": {
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available(),
        },
        "manifest_sha256": manifest_checksums,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(metadata, sort_keys=False))
    temporary.replace(path)


def _validate_resume(
    checkpoint: dict[str, object],
    config: ExperimentConfig,
    manifest_checksums: dict[str, str],
) -> None:
    saved_config = checkpoint["config"]
    if not isinstance(saved_config, dict):
        raise ValueError("checkpoint config is invalid")
    saved_values = to_dict(config_from_dict(saved_config))
    current_values = to_dict(config)
    if saved_values["model"] != current_values["model"]:
        raise ValueError("checkpoint model configuration does not match the current run")
    if (
        saved_values["train"]["scheduler"] == "cosine"
        and saved_values["train"]["epochs"] != current_values["train"]["epochs"]
    ):
        raise ValueError(
            "cannot change train.epochs when resuming a cosine schedule; "
            "restart the run or keep the checkpoint's planned total"
        )
    if checkpoint["label_map"] != config.data.label_map:
        raise ValueError("checkpoint label mapping does not match the current run")
    if checkpoint["ignore_index"] != config.loss.ignore_index:
        raise ValueError("checkpoint ignore index does not match the current run")
    if checkpoint["manifest_checksums"] != manifest_checksums:
        raise ValueError("checkpoint manifests do not match the current run")
    for values in (saved_values, current_values):
        values["data"] = dict(values["data"])
        values["data"].pop("data_dir")
        values["data"].pop("manifest_dir")
        values["data"].pop("label_map")
        values["data"].pop("ignore_values")
        values["loss"] = dict(values["loss"])
        values["loss"].pop("ignore_index")
        values["train"] = dict(values["train"])
        values["train"].pop("epochs")
    if any(saved_values[section] != current_values[section] for section in ("data", "loss", "train")):
        raise ValueError("checkpoint training configuration does not match the current run")


def train_from_config(
    config: ExperimentConfig,
    *,
    resume: str | Path | None = None,
    dry_run: bool = False,
) -> Path | None:
    """Run a dry check or a complete configured training job."""
    seed_everything(config.train.seed)
    device = resolve_device(config.device)
    manifest_dir = Path(config.data.manifest_dir)
    train_manifest = manifest_dir / "train.csv"
    valid_manifest = manifest_dir / "valid.csv"
    for path in (train_manifest, valid_manifest):
        if not path.is_file():
            raise FileNotFoundError(f"manifest not found: {path}")
    manifest_checksums = {"train": _sha256(train_manifest), "valid": _sha256(valid_manifest)}

    train_dataset = SegmentationDataset(
        train_manifest,
        build_transform(config.data, training=True),
        config.data.label_map,
        config.data.ignore_values,
        config.loss.ignore_index,
        config.data.max_train_samples,
    )
    valid_dataset = SegmentationDataset(
        valid_manifest,
        build_transform(config.data, training=False),
        config.data.label_map,
        config.data.ignore_values,
        config.loss.ignore_index,
        config.data.max_valid_samples,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        worker_init_fn=_seed_worker,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        worker_init_fn=_seed_worker,
    )

    model = create_model(
        config.model.name,
        in_channels=config.model.in_channels,
        num_classes=config.model.num_classes,
        base_channels=config.model.base_channels,
        pretrained=config.model.pretrained,
    ).to(device)
    criterion = SegmentationLoss(
        config.loss.cross_entropy_weight,
        config.loss.dice_weight,
        config.loss.ignore_index,
    )
    if config.train.optimizer != "adamw":
        raise ValueError(f"unsupported optimizer: {config.train.optimizer}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
    scheduler: LRScheduler | None
    if config.train.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=config.train.epochs)
    elif config.train.scheduler == "none":
        scheduler = None
    else:
        raise ValueError(f"unsupported scheduler: {config.train.scheduler}")

    if resume is not None and config.run_name is None:
        config.run_name = Path(resume).resolve().parent.name
    if config.run_name is None:
        config.run_name = f"{config.model.name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir = Path(config.output_dir) / config.run_name
    trainer = Trainer(model, criterion, optimizer, scheduler, device, config, run_dir, manifest_checksums)
    if dry_run:
        trainer.dry_run(train_loader)
        return None

    start_epoch = 1
    best_metric = -float("inf")
    if resume is None:
        if (run_dir / "best.pt").exists() or (run_dir / "last.pt").exists():
            raise ValueError(f"run directory already contains checkpoints: {run_dir}; use --resume")
    else:
        checkpoint_path = Path(resume).resolve()
        if run_dir.resolve() != checkpoint_path.parent:
            raise ValueError("resume checkpoint does not belong to the configured run directory")
        checkpoint = load_checkpoint(checkpoint_path)
        _validate_resume(checkpoint, config, manifest_checksums)
        completed_epoch, best_metric = restore_training_state(checkpoint, model, optimizer, scheduler)
        start_epoch = completed_epoch + 1
        if start_epoch > config.train.epochs:
            raise ValueError(
                f"checkpoint already completed epoch {completed_epoch}; train.epochs={config.train.epochs}"
            )

    run_dir.mkdir(parents=True, exist_ok=True)
    dump_config(config, run_dir / "config.yaml")
    _write_run_metadata(run_dir / "run.yaml", device, manifest_checksums)
    return trainer.fit(train_loader, valid_loader, start_epoch=start_epoch, best_metric=best_metric)

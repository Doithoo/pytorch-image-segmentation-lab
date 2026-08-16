"""Versioned, atomic checkpoint persistence."""

from __future__ import annotations

import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from .. import __version__
from ..config import ExperimentConfig, to_dict

CHECKPOINT_SCHEMA_VERSION = 1
CLASS_NAMES = ("background", "pet")
REQUIRED_KEYS = {
    "schema_version",
    "epoch",
    "best_metric",
    "model_state",
    "optimizer_state",
    "scheduler_state",
    "config",
    "class_names",
    "label_map",
    "ignore_index",
    "manifest_checksums",
    "model_metadata",
    "rng_state",
    "package_version",
}


def git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    *,
    epoch: int,
    best_metric: float,
    cfg: ExperimentConfig,
    manifest_checksums: dict[str, str],
) -> Path:
    """Atomically save all state required for exact training resume."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rng_state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_state["cuda"] = torch.cuda.get_rng_state_all()
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "epoch": epoch,
        "best_metric": best_metric,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "config": to_dict(cfg),
        "class_names": list(CLASS_NAMES),
        "label_map": dict(cfg.data.label_map),
        "ignore_index": cfg.loss.ignore_index,
        "manifest_checksums": dict(manifest_checksums),
        "model_metadata": dict(getattr(model, "checkpoint_metadata", {})),
        "rng_state": rng_state,
        "package_version": __version__,
        "git_revision": git_revision(),
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    return output


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load and validate a schema-one checkpoint on CPU."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(loaded, dict):
        raise ValueError("checkpoint root must be a mapping")
    schema = loaded.get("schema_version")
    if schema != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"checkpoint schema {schema} is not supported; expected {CHECKPOINT_SCHEMA_VERSION}")
    missing = sorted(REQUIRED_KEYS - set(loaded))
    if missing:
        raise ValueError(f"checkpoint is missing required keys: {missing}")
    return loaded


def restore_training_state(
    checkpoint: dict[str, Any],
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
) -> tuple[int, float]:
    """Restore model, optimizer, scheduler, and random number generators."""
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler_state = checkpoint["scheduler_state"]
    if scheduler_state is not None:
        if scheduler is None:
            raise ValueError("checkpoint contains scheduler state but the current run has no scheduler")
        scheduler.load_state_dict(scheduler_state)
    rng_state = checkpoint["rng_state"]
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch"])
    if "cuda" in rng_state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_state["cuda"])
    return int(checkpoint["epoch"]), float(checkpoint["best_metric"])

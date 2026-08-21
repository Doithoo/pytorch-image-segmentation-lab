"""Versioned, atomic checkpoint persistence."""

from __future__ import annotations

import hashlib
import json
import pickle
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
from ..data import ClassDefinition, LabelSchema
from .precision import PrecisionController

CHECKPOINT_SCHEMA_VERSION = 2
LEGACY_CHECKPOINT_SCHEMA_VERSION = 1
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


def _safe_numpy_rng_state() -> dict[str, Any]:
    """Encode NumPy's RNG state using only weights-only-safe values."""
    state = np.random.get_state()
    return {
        "algorithm": state[0],
        "state": torch.from_numpy(state[1].copy()),
        "position": int(state[2]),
        "has_gaussian": int(state[3]),
        "cached_gaussian": float(state[4]),
    }


def _restore_numpy_rng_state(state: object) -> None:
    if not isinstance(state, dict):
        raise ValueError("checkpoint NumPy RNG state is invalid")
    raw_state = state.get("state")
    if not isinstance(raw_state, torch.Tensor):
        raise ValueError("checkpoint NumPy RNG state array is invalid")
    np.random.set_state(
        (
            str(state["algorithm"]),
            raw_state.cpu().numpy().astype(np.uint32, copy=False),
            int(state["position"]),
            int(state["has_gaussian"]),
            float(state["cached_gaussian"]),
        )
    )


def _load_serialized_checkpoint(path: Path) -> dict[str, Any]:
    """Load a checkpoint without executing arbitrary pickle payloads."""
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except (RuntimeError, pickle.UnpicklingError) as exc:
        raise ValueError(
            "checkpoint is not weights-only safe; it may be a legacy or untrusted file. "
            "Recreate it with the current package before loading"
        ) from exc
    if not isinstance(loaded, dict):
        raise ValueError("checkpoint root must be a mapping")
    return loaded


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
    label_schema: LabelSchema | None = None,
    precision_state: dict[str, Any] | None = None,
) -> Path:
    """Atomically save all state required for exact training resume."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rng_state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": _safe_numpy_rng_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        rng_state["mps"] = torch.mps.get_rng_state()
    schema = label_schema or LabelSchema(
        classes=(
            ClassDefinition(0, "background", (32, 32, 32)),
            ClassDefinition(1, "pet", (46, 204, 113)),
        ),
        ignore_index=cfg.loss.ignore_index,
    )
    schema_hash = hashlib.sha256(
        json.dumps(schema.to_dict(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "epoch": epoch,
        "best_metric": best_metric,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "config": to_dict(cfg),
        "class_names": [item.name for item in schema.classes],
        "label_schema": schema.to_dict(),
        "label_schema_sha256": schema_hash,
        "precision_state": precision_state or {"requested": False, "effective": False, "scaler": None},
        "label_map": dict(cfg.data.label_map),
        "ignore_index": schema.ignore_index,
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
    """Load and validate a safe schema-two checkpoint on CPU."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    loaded = _load_serialized_checkpoint(checkpoint_path)
    schema = loaded.get("schema_version")
    if schema == LEGACY_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "legacy schema-one checkpoint is not weights-only safe; recreate it with a trusted older environment "
            "or retrain with the current package"
        )
    if schema != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"checkpoint schema {schema} is not supported; expected {CHECKPOINT_SCHEMA_VERSION}")
    missing = sorted(REQUIRED_KEYS - set(loaded))
    if missing:
        raise ValueError(f"checkpoint is missing required keys: {missing}")
    precision_state = loaded.get("precision_state")
    if (
        isinstance(precision_state, dict)
        and precision_state.get("effective") is True
        and not isinstance(precision_state.get("scaler"), dict)
    ):
        raise ValueError("effective CUDA AMP checkpoint is missing scaler state")
    return loaded


def checkpoint_label_schema(checkpoint: dict[str, Any]) -> LabelSchema:
    """Restore label semantics from new or legacy binary checkpoints."""
    raw_schema = checkpoint.get("label_schema")
    if raw_schema is not None:
        return LabelSchema.from_dict(raw_schema)
    names = checkpoint.get("class_names", CLASS_NAMES)
    colors = ((32, 32, 32), (46, 204, 113))
    classes = tuple(
        ClassDefinition(
            index,
            str(name),
            colors[index] if index < len(colors) else ((index * 67) % 256, (index * 131) % 256, (index * 197) % 256),
        )
        for index, name in enumerate(names)
    )
    return LabelSchema(classes=classes, ignore_index=int(checkpoint["ignore_index"]))


def restore_training_state(
    checkpoint: dict[str, Any],
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    precision: PrecisionController | None = None,
) -> tuple[int, float]:
    """Restore model, optimizer, scheduler, and random number generators."""
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler_state = checkpoint["scheduler_state"]
    if scheduler_state is not None:
        if scheduler is None:
            raise ValueError("checkpoint contains scheduler state but the current run has no scheduler")
        scheduler.load_state_dict(scheduler_state)
    if precision is not None:
        precision.load_state_dict(checkpoint.get("precision_state"))
    rng_state = checkpoint["rng_state"]
    random.setstate(rng_state["python"])
    _restore_numpy_rng_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch"])
    if "cuda" in rng_state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_state["cuda"])
    if "mps" in rng_state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(rng_state["mps"])
    return int(checkpoint["epoch"]), float(checkpoint["best_metric"])

"""CUDA mixed precision with explicit fallback and resumable scaler state."""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


class PrecisionController:
    """Own autocast, gradient scaling, and effective AMP semantics."""

    def __init__(self, device: torch.device, *, requested: bool) -> None:
        self.device = device
        self.requested = requested
        self.effective = requested and device.type == "cuda"
        if requested and not self.effective:
            warnings.warn(
                f"AMP requested on {device.type}; this project uses CUDA AMP only and will use full precision",
                UserWarning,
                stacklevel=2,
            )
        self.scaler = torch.amp.GradScaler("cuda", enabled=True) if self.effective else None

    def autocast(self) -> AbstractContextManager[Any]:
        """Return CUDA autocast or a no-op context."""
        if self.effective:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def backward_step(
        self,
        loss: torch.Tensor,
        optimizer: Optimizer,
        parameters: Iterable[nn.Parameter],
        *,
        grad_clip: float,
    ) -> None:
        """Backpropagate, optionally unscale and clip, then update parameters."""
        parameter_list = list(parameters)
        if self.scaler is None:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(parameter_list, grad_clip)
            optimizer.step()
            return
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(optimizer)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(parameter_list, grad_clip)
        self.scaler.step(optimizer)
        self.scaler.update()

    def state_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "effective": self.effective,
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
        }

    def load_state_dict(self, state: dict[str, Any] | None) -> None:
        """Restore scaler state while refusing changed effective semantics."""
        if state is None:
            if self.effective:
                raise ValueError("legacy checkpoint has no AMP state; resume it with train.amp=false")
            return
        saved_effective = state.get("effective")
        if saved_effective is not self.effective:
            raise ValueError(
                f"effective AMP semantics changed on resume: checkpoint={saved_effective}, current={self.effective}"
            )
        scaler_state = state.get("scaler")
        if self.effective:
            if self.scaler is None or not isinstance(scaler_state, dict):
                raise ValueError("effective CUDA AMP checkpoint is missing scaler state")
            self.scaler.load_state_dict(scaler_state)

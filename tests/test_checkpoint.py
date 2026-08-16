from pathlib import Path

import pytest
import torch

from image_segmenter.config import load_config
from image_segmenter.models import create_model
from image_segmenter.training.checkpoint import load_checkpoint, restore_training_state, save_checkpoint


def test_checkpoint_restores_model_optimizer_and_metadata(tmp_path: Path) -> None:
    cfg = load_config(overrides={"model.base_channels": 4, "data.image_size": 32})
    model = create_model("unet_small", in_channels=3, num_classes=2, base_channels=4, pretrained=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
    path = tmp_path / "last.pt"
    save_checkpoint(
        path,
        model,
        optimizer,
        None,
        epoch=3,
        best_metric=0.6,
        cfg=cfg,
        manifest_checksums={"train": "abc"},
    )
    loaded = load_checkpoint(path)
    assert loaded["schema_version"] == 1
    assert loaded["epoch"] == 3
    assert loaded["best_metric"] == 0.6
    assert loaded["config"]["data"]["label_map"] == {1: 1, 2: 0}
    assert loaded["model_metadata"] == {
        "architecture": "unet_small",
        "pretrained_weights": None,
        "provider": "image-segmenter",
        "parameters": {"base_channels": 4},
    }
    assert loaded["label_schema"]["classes"][1]["name"] == "pet"
    assert loaded["precision_state"] == {"requested": False, "effective": False, "scaler": None}
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    assert restore_training_state(loaded, model, optimizer, None) == (3, 0.6)
    assert all(torch.equal(model.state_dict()[name], value) for name, value in expected.items())


def test_checkpoint_restores_mps_rng_when_available(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config(overrides={"model.base_channels": 4, "data.image_size": 32})
    model = create_model("unet_small", in_channels=3, num_classes=2, base_channels=4, pretrained=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    expected_rng = torch.tensor([1, 2, 3], dtype=torch.uint8)
    restored_rng: list[torch.Tensor] = []
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "get_rng_state", lambda: expected_rng)
    monkeypatch.setattr(torch.mps, "set_rng_state", restored_rng.append)

    path = tmp_path / "last.pt"
    save_checkpoint(
        path,
        model,
        optimizer,
        None,
        epoch=1,
        best_metric=0.5,
        cfg=cfg,
        manifest_checksums={"train": "abc"},
    )
    loaded = load_checkpoint(path)
    restore_training_state(loaded, model, optimizer, None)

    assert torch.equal(loaded["rng_state"]["mps"], expected_rng)
    assert len(restored_rng) == 1
    assert torch.equal(restored_rng[0], expected_rng)


def test_unknown_checkpoint_schema_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "future.pt"
    torch.save({"schema_version": 99}, path)
    with pytest.raises(ValueError, match="checkpoint schema 99 is not supported"):
        load_checkpoint(path)


def test_incomplete_checkpoint_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "incomplete.pt"
    torch.save({"schema_version": 1}, path)
    with pytest.raises(ValueError, match="missing required keys"):
        load_checkpoint(path)


def test_effective_amp_checkpoint_requires_scaler_state(tmp_path: Path) -> None:
    cfg = load_config(overrides={"model.base_channels": 4, "data.image_size": 32})
    model = create_model("unet_small", in_channels=3, num_classes=2, base_channels=4, pretrained=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    path = save_checkpoint(
        tmp_path / "amp.pt",
        model,
        optimizer,
        None,
        epoch=1,
        best_metric=0.0,
        cfg=cfg,
        manifest_checksums={"train": "abc"},
        precision_state={"requested": True, "effective": True, "scaler": None},
    )

    with pytest.raises(ValueError, match="missing scaler state"):
        load_checkpoint(path)

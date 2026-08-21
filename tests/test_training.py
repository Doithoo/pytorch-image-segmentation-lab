import csv
from pathlib import Path

import pytest
import torch
import yaml

from image_segmenter.config import load_config
from image_segmenter.data.manifest import prepare_paired_manifests
from image_segmenter.training.checkpoint import load_checkpoint
from image_segmenter.training.train import train_from_config


def _config(paired_data: Path, tmp_path: Path):
    manifest_dir = tmp_path / "manifests"
    prepare_paired_manifests(
        paired_data,
        manifest_dir,
        [0.8, 0.1, 0.1],
        7,
        label_map={1: 1, 2: 0},
        ignore_values={3},
    )
    return load_config(
        overrides={
            "data.data_dir": str(paired_data),
            "data.manifest_dir": str(manifest_dir),
            "data.image_size": 32,
            "data.max_train_samples": 4,
            "data.max_valid_samples": 1,
            "model.base_channels": 4,
            "train.batch_size": 2,
            "train.epochs": 1,
            "device": "cpu",
            "output_dir": str(tmp_path / "artifacts"),
            "run_name": "tiny-run",
        }
    )


def test_dry_run_does_not_create_normal_artifacts(paired_data: Path, tmp_path: Path, capsys) -> None:
    cfg = _config(paired_data, tmp_path)
    result = train_from_config(cfg, dry_run=True)
    assert result is None
    output = capsys.readouterr().out
    assert "dry-run OK" in output
    assert "images: (2, 3, 32, 32)" in output
    assert not (Path(cfg.output_dir) / "tiny-run").exists()


def test_fit_writes_best_last_metrics_and_resumes(paired_data: Path, tmp_path: Path) -> None:
    cfg = _config(paired_data, tmp_path)
    run_dir = train_from_config(cfg)
    assert run_dir == Path(cfg.output_dir) / "tiny-run"
    assert (run_dir / "best.pt").is_file()
    assert (run_dir / "last.pt").is_file()
    assert (run_dir / "metrics.csv").read_text().count("\n") == 2
    assert b"\r\n" not in (run_dir / "metrics.csv").read_bytes()
    run_metadata = yaml.safe_load((run_dir / "run.yaml").read_text())
    assert run_metadata["schema_version"] == 1
    assert run_metadata["device"] == "cpu"
    assert run_metadata["python"]
    assert run_metadata["pytorch"]
    assert run_metadata["torchvision"]
    assert run_metadata["manifest_sha256"].keys() == {"train", "valid", "test"}
    assert run_metadata["parameter_count"] > 0
    assert run_metadata["trainable_parameter_count"] == run_metadata["parameter_count"]
    assert run_metadata["model"]["name"] == "unet_small"
    assert run_metadata["model"]["provider"] == "image-segmenter"
    assert run_metadata["model"]["factory"] is None
    assert run_metadata["effective_amp"] is False
    assert run_metadata["label_schema_sha256"]
    assert run_metadata["selected_metric"] == "class_iou:pet"
    assert run_metadata["total_training_seconds"] >= 0
    assert len(run_metadata["fit_history"]) == 1
    assert run_metadata["fit_history"][0]["device"] == "cpu"
    assert run_metadata["fit_history"][0]["start_epoch"] == 1
    assert run_metadata["fit_history"][0]["completed_epoch"] == 1
    with (run_dir / "metrics.csv").open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert float(row["epoch_duration_seconds"]) >= 0
    assert row["selected_metric"] == "class_iou:pet"
    assert load_checkpoint(run_dir / "last.pt")["epoch"] == 1
    cfg.train.epochs = 2
    train_from_config(cfg, resume=run_dir / "last.pt")
    assert (run_dir / "metrics.csv").read_text().count("\n") == 3
    assert load_checkpoint(run_dir / "last.pt")["epoch"] == 2
    resumed_metadata = yaml.safe_load((run_dir / "run.yaml").read_text())
    assert [event["start_epoch"] for event in resumed_metadata["fit_history"]] == [1, 2]
    assert [event["completed_epoch"] for event in resumed_metadata["fit_history"]] == [1, 2]


def test_resume_discards_metrics_ahead_of_last_checkpoint(paired_data: Path, tmp_path: Path) -> None:
    cfg = _config(paired_data, tmp_path)
    run_dir = train_from_config(cfg)
    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    interrupted_row = dict(rows[-1])
    interrupted_row["epoch"] = "2"
    with metrics_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writerow(interrupted_row)

    cfg.train.epochs = 2
    train_from_config(cfg, resume=run_dir / "last.pt")

    with metrics_path.open(newline="") as handle:
        resumed_rows = list(csv.DictReader(handle))
    assert [int(row["epoch"]) for row in resumed_rows] == [1, 2]


def test_existing_run_requires_resume(paired_data: Path, tmp_path: Path) -> None:
    cfg = _config(paired_data, tmp_path)
    train_from_config(cfg)
    try:
        train_from_config(cfg)
    except ValueError as exc:
        assert "already contains checkpoints" in str(exc)
    else:
        raise AssertionError("expected an existing run to be rejected")


def test_resume_matches_uninterrupted_training(paired_data: Path, tmp_path: Path) -> None:
    uninterrupted = _config(paired_data, tmp_path)
    uninterrupted.run_name = "uninterrupted"
    uninterrupted.train.epochs = 2
    uninterrupted_dir = train_from_config(uninterrupted)

    resumed = _config(paired_data, tmp_path)
    resumed.run_name = "resumed"
    resumed_dir = train_from_config(resumed)
    resumed.train.epochs = 2
    train_from_config(resumed, resume=resumed_dir / "last.pt")

    uninterrupted_state = load_checkpoint(uninterrupted_dir / "last.pt")["model_state"]
    resumed_state = load_checkpoint(resumed_dir / "last.pt")["model_state"]
    assert uninterrupted_state.keys() == resumed_state.keys()
    for name in uninterrupted_state:
        assert torch.equal(uninterrupted_state[name], resumed_state[name]), name


def test_resume_rejects_changed_training_semantics(paired_data: Path, tmp_path: Path) -> None:
    cfg = _config(paired_data, tmp_path)
    run_dir = train_from_config(cfg)
    cfg.train.epochs = 2
    cfg.train.batch_size = 1

    with pytest.raises(ValueError, match="training configuration"):
        train_from_config(cfg, resume=run_dir / "last.pt")


def test_resume_rejects_extending_a_cosine_schedule(paired_data: Path, tmp_path: Path) -> None:
    cfg = _config(paired_data, tmp_path)
    cfg.train.scheduler = "cosine"
    run_dir = train_from_config(cfg)
    cfg.train.epochs = 2

    with pytest.raises(ValueError, match="cosine schedule"):
        train_from_config(cfg, resume=run_dir / "last.pt")

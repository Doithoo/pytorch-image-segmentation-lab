from pathlib import Path

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
    assert load_checkpoint(run_dir / "last.pt")["epoch"] == 1
    cfg.train.epochs = 2
    train_from_config(cfg, resume=run_dir / "last.pt")
    assert (run_dir / "metrics.csv").read_text().count("\n") == 3
    assert load_checkpoint(run_dir / "last.pt")["epoch"] == 2


def test_existing_run_requires_resume(paired_data: Path, tmp_path: Path) -> None:
    cfg = _config(paired_data, tmp_path)
    train_from_config(cfg)
    try:
        train_from_config(cfg)
    except ValueError as exc:
        assert "already contains checkpoints" in str(exc)
    else:
        raise AssertionError("expected an existing run to be rejected")

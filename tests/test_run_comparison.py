import csv
from pathlib import Path

import pytest
import yaml

from image_segmenter.cli import main
from image_segmenter.evaluation.comparison import (
    compare_runs,
    format_comparison,
    read_run_artifacts,
    write_comparison_csv,
)


def _write_run(
    root: Path,
    name: str,
    *,
    model: str,
    learning_rate: float,
    schema_hash: str,
    metric_values: list[float],
) -> Path:
    run_dir = root / name
    run_dir.mkdir()
    config = {
        "config_version": 2,
        "model": {"name": model, "factory": None, "params": {}},
        "train": {"lr": learning_rate, "best_metric": "mean_iou"},
        "run_name": name,
    }
    metadata = {
        "schema_version": 1,
        "model": {"name": model, "provider": "test", "factory": None},
        "label_schema_sha256": schema_hash,
        "parameter_count": 100,
        "trainable_parameter_count": 100,
        "total_training_seconds": 2.5,
        "selected_metric": "mean_iou",
        "device": "cpu",
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (run_dir / "run.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False))
    with (run_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "valid_mean_iou", "valid_loss", "epoch_duration_seconds"],
            lineterminator="\n",
        )
        writer.writeheader()
        for epoch, value in enumerate(metric_values, start=1):
            writer.writerow(
                {
                    "epoch": epoch,
                    "valid_mean_iou": value,
                    "valid_loss": 1.0 - value,
                    "epoch_duration_seconds": 1.0,
                }
            )
    return run_dir


def test_compare_runs_selects_peak_epoch_and_sorts_metric(tmp_path: Path) -> None:
    first = _write_run(
        tmp_path,
        "unet-run",
        model="unet_small",
        learning_rate=0.001,
        schema_hash="same-schema",
        metric_values=[0.40, 0.55],
    )
    second = _write_run(
        tmp_path,
        "fcn-run",
        model="fcn_resnet50",
        learning_rate=0.0001,
        schema_hash="same-schema",
        metric_values=[0.65, 0.60],
    )

    report = compare_runs([first, second], metric="valid_mean_iou")

    assert [row.run_name for row in report.rows] == ["fcn-run", "unet-run"]
    assert [(row.epoch, row.metric_value) for row in report.rows] == [(1, 0.65), (2, 0.55)]
    assert report.comparable is True
    assert report.config_differences["model.name"] == ["unet_small", "fcn_resnet50"]
    assert report.config_differences["train.lr"] == [0.001, 0.0001]
    assert "\n..." not in format_comparison(report)


def test_loss_metrics_sort_lowest_value_first(tmp_path: Path) -> None:
    first = _write_run(
        tmp_path,
        "first",
        model="unet_small",
        learning_rate=0.001,
        schema_hash="same-schema",
        metric_values=[0.25],
    )
    second = _write_run(
        tmp_path,
        "second",
        model="unet_small",
        learning_rate=0.001,
        schema_hash="same-schema",
        metric_values=[0.75],
    )

    report = compare_runs([first, second], metric="valid_loss")

    assert [row.run_name for row in report.rows] == ["second", "first"]
    assert [row.metric_value for row in report.rows] == [0.25, 0.75]


@pytest.mark.parametrize("missing_name", ["config.yaml", "run.yaml", "metrics.csv"])
def test_read_run_artifacts_requires_all_files(tmp_path: Path, missing_name: str) -> None:
    run_dir = _write_run(
        tmp_path,
        "incomplete",
        model="unet_small",
        learning_rate=0.001,
        schema_hash="schema",
        metric_values=[0.5],
    )
    (run_dir / missing_name).unlink()

    with pytest.raises(FileNotFoundError, match=missing_name):
        read_run_artifacts(run_dir)


def test_compare_runs_rejects_different_label_schemas_by_default(tmp_path: Path) -> None:
    first = _write_run(
        tmp_path,
        "binary",
        model="unet_small",
        learning_rate=0.001,
        schema_hash="binary-schema",
        metric_values=[0.5],
    )
    second = _write_run(
        tmp_path,
        "multiclass",
        model="fcn_resnet50",
        learning_rate=0.001,
        schema_hash="multiclass-schema",
        metric_values=[0.6],
    )

    with pytest.raises(ValueError, match="label schemas differ"):
        compare_runs([first, second], metric="valid_mean_iou")

    report = compare_runs(
        [first, second],
        metric="valid_mean_iou",
        allow_different_schemas=True,
    )
    rendered = format_comparison(report)
    assert report.comparable is False
    assert "not directly comparable" in rendered
    assert "best model" not in rendered.lower()


def test_compare_runs_reports_missing_metric(tmp_path: Path) -> None:
    run_dir = _write_run(
        tmp_path,
        "run",
        model="unet_small",
        learning_rate=0.001,
        schema_hash="schema",
        metric_values=[0.5],
    )

    with pytest.raises(ValueError, match="valid_dice.*run"):
        compare_runs([run_dir], metric="valid_dice")


def test_comparison_csv_and_cli_preserve_sorted_rows(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    first = _write_run(
        tmp_path,
        "lower",
        model="unet_small",
        learning_rate=0.001,
        schema_hash="schema",
        metric_values=[0.4],
    )
    second = _write_run(
        tmp_path,
        "higher",
        model="fcn_resnet50",
        learning_rate=0.001,
        schema_hash="schema",
        metric_values=[0.7],
    )
    output = tmp_path / "comparison.csv"

    report = compare_runs([first, second], metric="valid_mean_iou")
    assert write_comparison_csv(report, output) == output
    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["run_name"] for row in rows] == ["higher", "lower"]
    assert rows[0]["metric"] == "valid_mean_iou"
    assert rows[0]["schema_comparable"] == "true"

    assert (
        main(
            [
                "compare-runs",
                str(first),
                str(second),
                "--metric",
                "valid_mean_iou",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    rendered = capsys.readouterr().out
    assert rendered.index("higher") < rendered.index("lower")
    assert "config differences:" in rendered
    assert f"comparison CSV: {output}" in rendered
    assert "best model" not in rendered.lower()

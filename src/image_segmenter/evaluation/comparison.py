"""Read-only comparison helpers for recorded training runs."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RunArtifacts:
    """Persisted configuration, metadata, and epoch metrics for one run."""

    run_dir: Path
    config: dict[str, Any]
    metadata: dict[str, Any]
    metrics: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class ComparisonRow:
    """One run's selected epoch for a requested metric."""

    run_dir: Path
    run_name: str
    model_name: str
    model_provider: str
    epoch: int
    metric: str
    metric_value: float
    parameter_count: int | None
    total_training_seconds: float | None
    device: str


@dataclass(frozen=True)
class ComparisonReport:
    """Sorted run rows plus factual configuration differences."""

    metric: str
    rows: tuple[ComparisonRow, ...]
    comparable: bool
    config_differences: dict[str, list[Any]]


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"run artifact not found: {path}")
    loaded = yaml.safe_load(path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError(f"run artifact must contain a YAML mapping: {path}")
    return loaded


def read_run_artifacts(run_dir: str | Path) -> RunArtifacts:
    """Read and validate the three portable artifacts in one run directory."""
    directory = Path(run_dir)
    config = _read_yaml_mapping(directory / "config.yaml")
    metadata = _read_yaml_mapping(directory / "run.yaml")
    metrics_path = directory / "metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"run artifact not found: {metrics_path}")
    with metrics_path.open(newline="") as handle:
        metrics = tuple(csv.DictReader(handle))
    if not metrics:
        raise ValueError(f"training metrics contain no epochs: {metrics_path}")
    return RunArtifacts(directory, config, metadata, metrics)


def _flatten(mapping: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in mapping.items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(_flatten(value, dotted))
        else:
            flattened[dotted] = value
    return flattened


def _config_differences(runs: list[RunArtifacts]) -> dict[str, list[Any]]:
    flattened = [_flatten(run.config) for run in runs]
    keys = sorted({key for config in flattened for key in config})
    differences: dict[str, list[Any]] = {}
    for key in keys:
        values = [config.get(key) for config in flattened]
        if any(value != values[0] for value in values[1:]):
            differences[key] = values
    return differences


def _number(value: object, *, field: str, run_dir: Path) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not numeric in run: {run_dir}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} is not finite in run: {run_dir}")
    return number


def _optional_number(value: object, *, field: str, run_dir: Path) -> float | None:
    if value is None:
        return None
    return _number(value, field=field, run_dir=run_dir)


def _comparison_row(run: RunArtifacts, metric: str, *, lower_is_better: bool) -> ComparisonRow:
    if metric not in run.metrics[0]:
        raise ValueError(f"metric {metric!r} is not recorded by run: {run.run_dir}")
    candidates: list[tuple[float, dict[str, str]]] = []
    for epoch_row in run.metrics:
        candidates.append((_number(epoch_row.get(metric), field=metric, run_dir=run.run_dir), epoch_row))
    value, selected = (min if lower_is_better else max)(candidates, key=lambda item: item[0])
    try:
        epoch = int(selected["epoch"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"epoch is not an integer in run: {run.run_dir}") from exc

    model = run.metadata.get("model", {})
    if not isinstance(model, dict):
        raise ValueError(f"model metadata is invalid in run: {run.run_dir}")
    configured_name = run.config.get("run_name")
    run_name = configured_name if isinstance(configured_name, str) and configured_name else run.run_dir.name
    parameter_count_value = _optional_number(
        run.metadata.get("parameter_count"), field="parameter_count", run_dir=run.run_dir
    )
    runtime_value = _optional_number(
        run.metadata.get("total_training_seconds"),
        field="total_training_seconds",
        run_dir=run.run_dir,
    )
    return ComparisonRow(
        run_dir=run.run_dir,
        run_name=run_name,
        model_name=str(model.get("name", "unknown")),
        model_provider=str(model.get("provider", "unknown")),
        epoch=epoch,
        metric=metric,
        metric_value=value,
        parameter_count=int(parameter_count_value) if parameter_count_value is not None else None,
        total_training_seconds=runtime_value,
        device=str(run.metadata.get("device", "unknown")),
    )


def compare_runs(
    run_dirs: list[str | Path],
    *,
    metric: str,
    allow_different_schemas: bool = False,
) -> ComparisonReport:
    """Compare one recorded metric without claiming hardware-normalized quality."""
    if not run_dirs:
        raise ValueError("at least one run directory is required")
    if not metric:
        raise ValueError("metric must be a non-empty string")
    runs = [read_run_artifacts(path) for path in run_dirs]
    schema_hashes = {str(run.metadata.get("label_schema_sha256", "")) for run in runs}
    if "" in schema_hashes:
        raise ValueError("label schema hash is missing from one or more runs")
    comparable = len(schema_hashes) == 1
    if not comparable and not allow_different_schemas:
        raise ValueError("run label schemas differ; pass --allow-different-schemas to inspect them")

    lower_is_better = "loss" in metric.lower()
    rows = [_comparison_row(run, metric, lower_is_better=lower_is_better) for run in runs]
    rows.sort(key=lambda row: (row.metric_value if lower_is_better else -row.metric_value, row.run_name))
    return ComparisonReport(
        metric=metric,
        rows=tuple(rows),
        comparable=comparable,
        config_differences=_config_differences(runs),
    )


def format_comparison(report: ComparisonReport) -> str:
    """Render a compact factual table and the differing configuration values."""
    lines = [f"metric: {report.metric}"]
    if report.comparable:
        lines.append("schema status: compatible")
    else:
        lines.append("schema status: different; metric values are not directly comparable")
    lines.append("run\tmodel\tepoch\tvalue\tparameters\ttraining_seconds\tdevice")
    for row in report.rows:
        parameters = "" if row.parameter_count is None else str(row.parameter_count)
        runtime = "" if row.total_training_seconds is None else f"{row.total_training_seconds:.6g}"
        lines.append(
            f"{row.run_name}\t{row.model_name}\t{row.epoch}\t{row.metric_value:.6g}\t"
            f"{parameters}\t{runtime}\t{row.device}"
        )
    lines.append("config differences:")
    if not report.config_differences:
        lines.append("  none")
    else:
        for key, values in report.config_differences.items():
            rendered = ", ".join(json.dumps(value, sort_keys=True) for value in values)
            lines.append(f"  {key}: {rendered}")
    return "\n".join(lines) + "\n"


def write_comparison_csv(report: ComparisonReport, output: str | Path) -> Path:
    """Write the sorted rows to a portable CSV table."""
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_name",
        "run_dir",
        "model_name",
        "model_provider",
        "epoch",
        "metric",
        "metric_value",
        "parameter_count",
        "total_training_seconds",
        "device",
        "schema_comparable",
    ]
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in report.rows:
            writer.writerow(
                {
                    "run_name": row.run_name,
                    "run_dir": row.run_dir.as_posix(),
                    "model_name": row.model_name,
                    "model_provider": row.model_provider,
                    "epoch": row.epoch,
                    "metric": row.metric,
                    "metric_value": row.metric_value,
                    "parameter_count": "" if row.parameter_count is None else row.parameter_count,
                    "total_training_seconds": (
                        "" if row.total_training_seconds is None else row.total_training_seconds
                    ),
                    "device": row.device,
                    "schema_comparable": str(report.comparable).lower(),
                }
            )
    return destination

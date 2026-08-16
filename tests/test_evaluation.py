import csv
import json
from pathlib import Path

from PIL import Image

from image_segmenter.evaluation.evaluate import evaluate_checkpoint


def test_evaluate_checkpoint_writes_numeric_and_visual_evidence(trained_run: dict[str, Path]) -> None:
    output = evaluate_checkpoint(
        trained_run["checkpoint"],
        split="valid",
        device="cpu",
        sample_limit=1,
        error_limit=1,
    )
    metrics_text = (output / "metrics.json").read_text()
    metrics = json.loads(metrics_text)
    assert "NaN" not in metrics_text
    assert metrics["split"] == "valid"
    assert "pet_iou" in metrics["global"]
    with (output / "per_class_metrics.csv").open(newline="") as handle:
        assert {row["class_name"] for row in csv.DictReader(handle)} == {"background", "pet"}
    with (output / "per_image_metrics.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 1
    sample_files = list((output / "sample_overlays").glob("*.png"))
    worst_files = list((output / "worst_cases").glob("*.png"))
    assert len(sample_files) == 1
    assert len(worst_files) == 1
    assert Image.open(sample_files[0]).size == (32 * 4, 32 + 24)


def test_evaluation_records_relative_checkpoint_path(
    trained_run: dict[str, Path], monkeypatch
) -> None:
    checkpoint = trained_run["checkpoint"]
    working_directory = checkpoint.parents[2]
    monkeypatch.chdir(working_directory)
    relative_checkpoint = checkpoint.relative_to(working_directory)

    output = evaluate_checkpoint(
        relative_checkpoint,
        split="valid",
        device="cpu",
        sample_limit=0,
        error_limit=0,
    )

    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["checkpoint"] == relative_checkpoint.as_posix()

import csv
import hashlib
import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from image_segmenter.evaluation.evaluate import evaluate_checkpoint
from image_segmenter.training.checkpoint import load_checkpoint


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
    for filename in ("per_class_metrics.csv", "per_image_metrics.csv"):
        assert b"\r\n" not in (output / filename).read_bytes()
    with (output / "per_class_metrics.csv").open(newline="") as handle:
        assert {row["class_name"] for row in csv.DictReader(handle)} == {"background", "pet"}
    with (output / "per_image_metrics.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 1
    sample_files = list((output / "sample_overlays").glob("*.png"))
    worst_files = list((output / "worst_cases").glob("*.png"))
    assert len(sample_files) == 1
    assert len(worst_files) == 1
    assert Image.open(sample_files[0]).size == (32 * 4, 32 + 24)


def test_evaluation_records_relative_checkpoint_path(trained_run: dict[str, Path], monkeypatch) -> None:
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


def test_evaluation_rejects_manifest_changed_since_training(trained_run: dict[str, Path]) -> None:
    saved = load_checkpoint(trained_run["checkpoint"])
    manifest_dir = Path(saved["config"]["data"]["manifest_dir"])
    metadata_path = manifest_dir / "dataset.yaml"
    metadata = yaml.safe_load(metadata_path.read_text())
    valid_manifest = manifest_dir / metadata["manifests"]["valid"]
    valid_manifest.write_text(valid_manifest.read_text() + "\n")
    metadata["manifest_sha256"]["valid"] = hashlib.sha256(valid_manifest.read_bytes()).hexdigest()
    metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=False))

    with pytest.raises(ValueError, match="checkpoint manifest hash.*valid"):
        evaluate_checkpoint(trained_run["checkpoint"], split="valid", device="cpu")


def test_evaluation_removes_stale_visual_outputs(trained_run: dict[str, Path], tmp_path: Path) -> None:
    output = tmp_path / "evaluation"
    for folder in ("sample_overlays", "worst_cases"):
        destination = output / folder
        destination.mkdir(parents=True)
        (destination / "stale.png").write_bytes(b"stale")
    (output / "loss_curve.png").write_bytes(b"stale")

    evaluate_checkpoint(
        trained_run["checkpoint"],
        split="valid",
        device="cpu",
        output_dir=output,
        sample_limit=0,
        error_limit=0,
        plot=False,
    )

    assert not list((output / "sample_overlays").glob("*.png"))
    assert not list((output / "worst_cases").glob("*.png"))
    assert not (output / "loss_curve.png").exists()

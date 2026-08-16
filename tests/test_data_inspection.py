from pathlib import Path

import pytest
from PIL import Image

from image_segmenter.cli import main
from image_segmenter.data import inspect_prepared_data
from image_segmenter.data.manifest import prepare_paired_manifests


def test_inspection_reports_values_ratios_schema_and_preview(paired_data: Path, tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    prepare_paired_manifests(
        paired_data,
        prepared,
        [0.8, 0.1, 0.1],
        seed=7,
        label_map={1: 1, 2: 0},
        ignore_values={3},
    )
    preview = tmp_path / "preview.png"

    report = inspect_prepared_data(prepared, split="train", limit=2, preview=preview)

    assert report["split_counts"] == {"train": 8, "valid": 1, "test": 1}
    assert report["inspected_samples"] == 2
    assert report["raw_values"] == [1, 2, 3]
    assert report["mapped_values"] == [0, 1, 255]
    assert report["class_ratios"]["background"] > report["class_ratios"]["pet"]
    assert report["ignore_ratio"] > 0
    assert report["classes"][1] == {"id": 1, "name": "pet", "color": [46, 204, 113]}
    with Image.open(preview) as rendered:
        assert rendered.size == (32 * 4, (32 + 24) * 2)


@pytest.mark.parametrize("limit", [0, -1])
def test_inspection_rejects_nonpositive_limit(paired_data: Path, tmp_path: Path, limit: int) -> None:
    prepared = tmp_path / "prepared"
    prepare_paired_manifests(
        paired_data,
        prepared,
        [0.8, 0.1, 0.1],
        seed=7,
        label_map={1: 1, 2: 0},
        ignore_values={3},
    )
    with pytest.raises(ValueError, match="limit must be positive"):
        inspect_prepared_data(prepared, limit=limit)


def test_inspect_data_cli_prints_report(paired_data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    prepared = tmp_path / "prepared"
    prepare_paired_manifests(
        paired_data,
        prepared,
        [0.8, 0.1, 0.1],
        seed=7,
        label_map={1: 1, 2: 0},
        ignore_values={3},
    )

    assert main(["inspect-data", "--manifest-dir", str(prepared), "--limit", "1"]) == 0
    output = capsys.readouterr().out
    assert "provider: paired_masks" in output
    assert "raw_values:" in output

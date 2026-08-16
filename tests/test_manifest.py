import csv
from pathlib import Path

import pytest
from PIL import Image

from image_segmenter.data.manifest import ManifestError, prepare_oxford_manifests, prepare_paired_manifests


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_prepare_pairs_and_splits_deterministically(paired_data: Path, tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    prepare_paired_manifests(paired_data, first, [0.8, 0.1, 0.1], seed=7, label_map={1: 1, 2: 0}, ignore_values={3})
    prepare_paired_manifests(paired_data, second, [0.8, 0.1, 0.1], seed=7, label_map={1: 1, 2: 0}, ignore_values={3})
    assert [len(_rows(first / f"{split}.csv")) for split in ("train", "valid", "test")] == [8, 1, 1]
    assert [row["sample_id"] for row in _rows(first / "train.csv")] == [
        row["sample_id"] for row in _rows(second / "train.csv")
    ]
    assert b"\r\n" not in (first / "train.csv").read_bytes()


def test_prepare_rejects_ratios_that_produce_an_empty_split(paired_data: Path, tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="empty split"):
        prepare_paired_manifests(
            paired_data,
            tmp_path / "manifests",
            [0.8, 0.2, 0.0],
            seed=7,
            label_map={1: 1, 2: 0},
            ignore_values={3},
        )


def test_missing_mask_is_reported_with_invalid_csv(paired_data: Path, tmp_path: Path) -> None:
    (paired_data / "masks" / "sample_00.png").unlink()
    output = tmp_path / "manifests"
    with pytest.raises(ManifestError, match="1 invalid sample"):
        prepare_paired_manifests(
            paired_data, output, [0.8, 0.1, 0.1], seed=7, label_map={1: 1, 2: 0}, ignore_values={3}
        )
    assert "missing mask" in (output / "invalid.csv").read_text()


def test_empty_foreground_warns_but_is_retained(paired_data: Path, tmp_path: Path) -> None:
    Image.new("L", (32, 32), color=2).save(paired_data / "masks" / "sample_00.png")
    with pytest.warns(UserWarning, match="1 sample has no foreground"):
        paths = prepare_paired_manifests(
            paired_data,
            tmp_path / "manifests",
            [0.8, 0.1, 0.1],
            seed=7,
            label_map={1: 1, 2: 0},
            ignore_values={3},
        )
    assert sum(len(_rows(path)) for path in paths.values()) == 10


def test_manifest_paths_resolve_from_manifest_directory(paired_data: Path, tmp_path: Path) -> None:
    output = tmp_path / "nested" / "manifests"
    paths = prepare_paired_manifests(
        paired_data, output, [0.8, 0.1, 0.1], seed=7, label_map={1: 1, 2: 0}, ignore_values={3}
    )
    row = _rows(paths["train"])[0]
    assert (output / row["image_path"]).resolve().is_file()
    assert (output / row["mask_path"]).resolve().is_file()


def test_oxford_split_is_stratified_and_official_test_is_untouched(oxford_data: Path, tmp_path: Path) -> None:
    output = tmp_path / "manifests"
    paths = prepare_oxford_manifests(oxford_data, output, seed=11, label_map={1: 1, 2: 0}, ignore_values={3})
    train_rows = _rows(paths["train"])
    valid_rows = _rows(paths["valid"])
    test_rows = _rows(paths["test"])
    assert len(train_rows) == 6
    assert len(valid_rows) == 2
    assert {row["sample_id"] for row in test_rows} == {"breed_a_04", "breed_b_04"}
    assert {row["breed"] for row in train_rows} == {"breed_a", "breed_b"}
    assert {row["source_split"] for row in test_rows} == {"test"}
    assert (output / "source.yaml").is_file()
    assert "train: 6" in (output / "summary.txt").read_text()


def test_oxford_unlisted_archive_pairs_are_audited_and_excluded(oxford_data: Path, tmp_path: Path) -> None:
    Image.new("RGB", (24, 24), color=(20, 40, 60)).save(oxford_data / "images" / "extra_breed_99.jpg")
    Image.new("L", (24, 24), color=1).save(oxford_data / "masks" / "extra_breed_99.png")
    output = tmp_path / "manifests"

    paths = prepare_oxford_manifests(oxford_data, output, seed=11, label_map={1: 1, 2: 0}, ignore_values={3})

    listed = {row["sample_id"] for path in paths.values() for row in _rows(path)}
    assert "extra_breed_99" not in listed
    assert "unlisted_pair_count: 1" in (output / "source.yaml").read_text()
    assert "unlisted_pair_count: 1" in (output / "summary.txt").read_text()

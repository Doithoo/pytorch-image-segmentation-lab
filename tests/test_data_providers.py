import csv
import shutil
from pathlib import Path

import pytest

from image_segmenter.data import ManifestError, prepare_dataset, read_dataset_metadata


def _read_ids(path: Path) -> set[str]:
    with path.open(newline="") as handle:
        return {row["sample_id"] for row in csv.DictReader(handle)}


def test_paired_provider_preserves_existing_wrapper_behavior(paired_data: Path, tmp_path: Path) -> None:
    paths = prepare_dataset(
        "paired_masks",
        data_dir=paired_data,
        manifest_dir=tmp_path / "prepared",
        split_ratios=[0.8, 0.1, 0.1],
        seed=7,
        label_map={1: 1, 2: 0},
        ignore_values={3},
        params={},
    )

    assert [_read_ids(paths[split]) for split in ("train", "valid", "test")]
    assert read_dataset_metadata(tmp_path / "prepared" / "dataset.yaml").provider == "paired_masks"


def test_paired_provider_keeps_groups_in_one_split(paired_data: Path, tmp_path: Path) -> None:
    source = tmp_path / "grouped"
    for index, image in enumerate(sorted((paired_data / "images").glob("*.png"))):
        group = f"group_{index // 2}"
        (source / "images" / group).mkdir(parents=True, exist_ok=True)
        (source / "masks" / group).mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, source / "images" / group / image.name)
        shutil.copy2(paired_data / "masks" / image.name, source / "masks" / group / image.name)

    paths = prepare_dataset(
        "paired_masks",
        data_dir=source,
        manifest_dir=tmp_path / "prepared",
        split_ratios=[0.6, 0.2, 0.2],
        seed=7,
        label_map={1: 1, 2: 0},
        ignore_values={3},
        params={"group_regex": r"(group_\d+)/"},
    )

    group_splits: dict[str, str] = {}
    for split, path in paths.items():
        for sample_id in _read_ids(path):
            group = sample_id.split("/", 1)[0]
            assert group_splits.setdefault(group, split) == split


def test_split_folders_preserves_user_membership(paired_data: Path, tmp_path: Path) -> None:
    source = tmp_path / "split-source"
    samples = sorted((paired_data / "images").glob("*.png"))
    expected: dict[str, set[str]] = {"train": set(), "valid": set(), "test": set()}
    for split, split_samples in zip(("train", "valid", "test"), (samples[:6], samples[6:8], samples[8:]), strict=True):
        (source / split / "images").mkdir(parents=True)
        (source / split / "masks").mkdir(parents=True)
        for image in split_samples:
            shutil.copy2(image, source / split / "images" / image.name)
            shutil.copy2(paired_data / "masks" / image.name, source / split / "masks" / image.name)
            expected[split].add(image.stem)

    paths = prepare_dataset(
        "split_folders",
        data_dir=source,
        manifest_dir=tmp_path / "prepared",
        split_ratios=[0.8, 0.1, 0.1],
        seed=7,
        label_map={1: 1, 2: 0},
        ignore_values={3},
        params={},
    )

    assert {split: _read_ids(path) for split, path in paths.items()} == expected


def test_manifest_provider_rejects_group_in_multiple_splits(paired_data: Path, tmp_path: Path) -> None:
    source = tmp_path / "source-manifests"
    source.mkdir()
    images = sorted((paired_data / "images").glob("*.png"))
    for split, image in zip(("train", "valid", "test"), images, strict=False):
        mask = paired_data / "masks" / image.name
        with (source / f"{split}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("sample_id", "image_path", "mask_path", "group_id"))
            writer.writeheader()
            writer.writerow(
                {
                    "sample_id": image.stem,
                    "image_path": image,
                    "mask_path": mask,
                    "group_id": "same-patient" if split != "test" else "other-patient",
                }
            )

    with pytest.raises(ManifestError, match="group.*same-patient.*multiple splits"):
        prepare_dataset(
            "manifest",
            data_dir=paired_data,
            manifest_dir=tmp_path / "prepared",
            split_ratios=[0.8, 0.1, 0.1],
            seed=7,
            label_map={1: 1, 2: 0},
            ignore_values={3},
            params={"manifests": {split: str(source / f"{split}.csv") for split in ("train", "valid", "test")}},
        )


def test_manifest_provider_validates_image_mask_dimensions(paired_data: Path, tmp_path: Path) -> None:
    source = tmp_path / "source-manifests"
    source.mkdir()
    images = sorted((paired_data / "images").glob("*.png"))
    manifests: dict[str, str] = {}
    for split, image in zip(("train", "valid", "test"), images, strict=False):
        path = source / f"{split}.csv"
        manifests[split] = str(path)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("sample_id", "image_path", "mask_path"))
            writer.writeheader()
            writer.writerow(
                {
                    "sample_id": image.stem,
                    "image_path": image,
                    "mask_path": paired_data / "masks" / image.name,
                }
            )

    paths = prepare_dataset(
        "manifest",
        data_dir=paired_data,
        manifest_dir=tmp_path / "prepared",
        split_ratios=[0.8, 0.1, 0.1],
        seed=7,
        label_map={1: 1, 2: 0},
        ignore_values={3},
        params={"manifests": manifests},
    )

    assert all(path.is_file() for path in paths.values())

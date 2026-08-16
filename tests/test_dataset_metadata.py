from pathlib import Path

import pytest
import yaml

from image_segmenter.data import (
    ClassDefinition,
    LabelSchema,
    PreparedDatasetMetadata,
    read_dataset_metadata,
    write_dataset_metadata,
)
from image_segmenter.data.manifest import prepare_paired_manifests


def _schema() -> LabelSchema:
    return LabelSchema(
        classes=(
            ClassDefinition(0, "background", (32, 32, 32)),
            ClassDefinition(1, "pet", (46, 204, 113)),
        ),
        ignore_index=255,
    )


def test_dataset_metadata_round_trips_with_portable_paths(tmp_path: Path) -> None:
    root = tmp_path / "prepared"
    root.mkdir()
    for split in ("train", "valid", "test"):
        (root / f"{split}.csv").write_text(f"sample_id\n{split}\n")
    metadata = PreparedDatasetMetadata.create(
        provider="paired_masks",
        split_method="random",
        seed=7,
        data_dir="../raw",
        manifests={split: f"{split}.csv" for split in ("train", "valid", "test")},
        split_counts={"train": 8, "valid": 1, "test": 1},
        label_schema=_schema(),
        root=root,
        label_map={1: 1, 2: 0},
        ignore_values=(3,),
    )

    path = write_dataset_metadata(root / "dataset.yaml", metadata)
    restored = read_dataset_metadata(path)

    assert restored == metadata
    assert all(not Path(value).is_absolute() for value in restored.manifests.values())
    assert not Path(restored.data_dir).is_absolute()
    assert not list(root.glob(".dataset.yaml.*.tmp"))


def test_dataset_metadata_rejects_unknown_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "dataset.yaml"
    path.write_text("schema_version: 99\n")

    with pytest.raises(ValueError, match="dataset metadata schema 99"):
        read_dataset_metadata(path)


def test_dataset_metadata_detects_tampered_manifest(tmp_path: Path) -> None:
    root = tmp_path / "prepared"
    root.mkdir()
    manifest = root / "train.csv"
    manifest.write_text("sample_id\na\n")
    metadata = PreparedDatasetMetadata.create(
        provider="manifest",
        split_method="provided",
        seed=None,
        data_dir="../raw",
        manifests={"train": "train.csv"},
        split_counts={"train": 1},
        label_schema=_schema(),
        root=root,
    )
    path = write_dataset_metadata(root / "dataset.yaml", metadata)
    manifest.write_text("sample_id\nb\n")

    with pytest.raises(ValueError, match="manifest hash mismatch.*train"):
        read_dataset_metadata(path)


def test_dataset_metadata_detects_tampered_label_schema(tmp_path: Path) -> None:
    root = tmp_path / "prepared"
    root.mkdir()
    (root / "train.csv").write_text("sample_id\na\n")
    metadata = PreparedDatasetMetadata.create(
        provider="manifest",
        split_method="provided",
        seed=None,
        data_dir="../raw",
        manifests={"train": "train.csv"},
        split_counts={"train": 1},
        label_schema=_schema(),
        root=root,
    )
    path = write_dataset_metadata(root / "dataset.yaml", metadata)
    raw = yaml.safe_load(path.read_text())
    raw["label_schema"]["classes"][1]["name"] = "changed"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))

    with pytest.raises(ValueError, match="label schema hash mismatch"):
        read_dataset_metadata(path)


def test_paired_preparation_emits_dataset_metadata(paired_data: Path, tmp_path: Path) -> None:
    output = tmp_path / "manifests"
    prepare_paired_manifests(
        paired_data,
        output,
        [0.8, 0.1, 0.1],
        seed=7,
        label_map={1: 1, 2: 0},
        ignore_values={3},
    )

    metadata = read_dataset_metadata(output / "dataset.yaml")
    assert metadata.provider == "paired_masks"
    assert metadata.split_method == "random"
    assert metadata.split_counts == {"train": 8, "valid": 1, "test": 1}
    assert metadata.label_schema.num_classes == 2

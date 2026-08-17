from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from image_segmenter.config import load_config
from image_segmenter.data import prepare_dataset, read_dataset_metadata


@pytest.fixture
def voc_data(tmp_path: Path) -> Path:
    root = tmp_path / "VOCdevkit" / "VOC2012"
    (root / "JPEGImages").mkdir(parents=True)
    (root / "SegmentationClass").mkdir()
    (root / "ImageSets" / "Segmentation").mkdir(parents=True)
    sample_ids = ["2007_000001", "2007_000002", "2007_000003"]
    for index, sample_id in enumerate(sample_ids):
        Image.new("RGB", (16, 12), color=(index * 20, 40, 80)).save(root / "JPEGImages" / f"{sample_id}.jpg")
        mask = np.zeros((12, 16), dtype=np.uint8)
        mask[:, 4:8] = index + 1
        mask[:, 8:12] = 255
        Image.fromarray(mask).save(root / "SegmentationClass" / f"{sample_id}.png")
    (root / "ImageSets" / "Segmentation" / "train.txt").write_text("2007_000001\n2007_000002\n")
    (root / "ImageSets" / "Segmentation" / "val.txt").write_text("2007_000003\n")
    return tmp_path


def test_voc_provider_uses_official_schema_and_membership(voc_data: Path, tmp_path: Path) -> None:
    output = tmp_path / "prepared"
    paths = prepare_dataset(
        "voc2012",
        data_dir=voc_data,
        manifest_dir=output,
        split_ratios=[0.8, 0.1, 0.1],
        seed=7,
        label_map={1: 1, 2: 0},
        ignore_values={3},
        params={"test_policy": "use_valid"},
    )

    metadata = read_dataset_metadata(output / "dataset.yaml")
    assert metadata.provider == "voc2012"
    assert metadata.label_schema.num_classes == 21
    assert metadata.label_schema.classes[0].name == "background"
    assert metadata.label_schema.classes[15].name == "person"
    assert metadata.label_schema.classes[1].color == (128, 0, 0)
    assert metadata.label_schema.ignore_index == 255
    assert len(paths["train"].read_text().splitlines()) - 1 == 2
    assert "2007_000003" in paths["valid"].read_text()
    assert "2007_000003" in paths["test"].read_text()


def test_voc_provider_requires_explicit_test_policy(voc_data: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="test_policy.*test_list"):
        prepare_dataset(
            "voc2012",
            data_dir=voc_data,
            manifest_dir=tmp_path / "prepared",
            split_ratios=[0.8, 0.1, 0.1],
            seed=7,
            label_map={1: 1, 2: 0},
            ignore_values={3},
            params={},
        )


def test_voc_provider_points_missing_data_to_guide(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="docs/reference/voc2012"):
        prepare_dataset(
            "voc2012",
            data_dir=tmp_path / "missing",
            manifest_dir=tmp_path / "prepared",
            split_ratios=[0.8, 0.1, 0.1],
            seed=7,
            label_map={1: 1, 2: 0},
            ignore_values={3},
            params={"test_policy": "use_valid"},
        )


def test_voc_example_config_loads_as_multiclass() -> None:
    config = load_config(Path(__file__).parents[1] / "configs" / "voc2012.yaml")
    assert config.data.name == "voc2012"
    assert config.model.expected_num_classes == 21
    assert config.train.best_metric == "mean_iou"

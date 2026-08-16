from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def paired_data(tmp_path: Path) -> Path:
    root = tmp_path / "paired"
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir(parents=True)
    for index in range(10):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[:, :, index % 3] = 80 + index
        mask = np.full((32, 32), 2, dtype=np.uint8)
        mask[8:24, 8:24] = 1
        mask[6:10, 6:26] = 3
        Image.fromarray(image).save(root / "images" / f"sample_{index:02d}.png")
        Image.fromarray(mask).save(root / "masks" / f"sample_{index:02d}.png")
    return root


@pytest.fixture
def oxford_data(tmp_path: Path) -> Path:
    root = tmp_path / "oxford"
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir(parents=True)
    (root / "annotations").mkdir(parents=True)
    trainval: list[str] = []
    test: list[str] = []
    for breed_index, breed in enumerate(("breed_a", "breed_b"), start=1):
        for image_index in range(5):
            sample_id = f"{breed}_{image_index:02d}"
            Image.new("RGB", (24, 24), color=(breed_index * 40, image_index * 10, 80)).save(
                root / "images" / f"{sample_id}.jpg"
            )
            mask = np.full((24, 24), 2, dtype=np.uint8)
            mask[5:19, 5:19] = 1
            mask[4, 4:20] = 3
            Image.fromarray(mask).save(root / "masks" / f"{sample_id}.png")
            line = f"{sample_id} {breed_index} 1 {breed_index}"
            (test if image_index == 4 else trainval).append(line)
    (root / "annotations" / "trainval.txt").write_text("\n".join(trainval) + "\n")
    (root / "annotations" / "test.txt").write_text("\n".join(test) + "\n")
    return root


@pytest.fixture
def trained_run(paired_data: Path, tmp_path: Path) -> dict[str, Path]:
    from image_segmenter.config import load_config
    from image_segmenter.data.manifest import prepare_paired_manifests
    from image_segmenter.training.train import train_from_config

    manifests = tmp_path / "manifests"
    prepare_paired_manifests(
        paired_data,
        manifests,
        [0.8, 0.1, 0.1],
        seed=7,
        label_map={1: 1, 2: 0},
        ignore_values={3},
    )
    artifacts = tmp_path / "artifacts"
    config = load_config(
        overrides={
            "data.manifest_dir": str(manifests),
            "data.image_size": 32,
            "data.max_train_samples": 4,
            "data.max_valid_samples": 1,
            "model.base_channels": 4,
            "train.batch_size": 2,
            "train.epochs": 1,
            "device": "cpu",
            "output_dir": str(artifacts),
            "run_name": "trained-fixture",
        }
    )
    run_dir = train_from_config(config)
    assert run_dir is not None
    return {
        "run_dir": run_dir,
        "checkpoint": run_dir / "best.pt",
        "image": sorted((paired_data / "images").glob("*.png"))[0],
    }

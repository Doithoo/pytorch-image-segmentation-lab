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

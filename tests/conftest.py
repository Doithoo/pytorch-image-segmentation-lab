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

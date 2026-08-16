from pathlib import Path

import pytest
import torch

from image_segmenter.data.dataset import SegmentationDataset, map_mask
from image_segmenter.data.manifest import prepare_paired_manifests
from image_segmenter.data.transforms import PairedTransform


def test_map_mask_preserves_classes_and_marks_ignore() -> None:
    raw = torch.tensor([[1, 2, 3]], dtype=torch.int64)
    mapped = map_mask(raw, {1: 1, 2: 0}, [3], ignore_index=255)
    assert mapped.tolist() == [[1, 0, 255]]


def test_map_mask_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match=r"unmapped values: \[9\]"):
        map_mask(torch.tensor([[1, 9]]), {1: 1, 2: 0}, [3], ignore_index=255)


def test_dataset_returns_aligned_tensor_contract(paired_data: Path, tmp_path: Path) -> None:
    manifests = prepare_paired_manifests(
        paired_data,
        tmp_path / "manifests",
        [0.8, 0.1, 0.1],
        seed=7,
        label_map={1: 1, 2: 0},
        ignore_values={3},
    )
    transform = PairedTransform(16, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], horizontal_flip=1.0)
    dataset = SegmentationDataset(manifests["train"], transform, {1: 1, 2: 0}, [3], 255)
    image, mask, sample_id = dataset[0]
    assert image.shape == (3, 16, 16)
    assert image.dtype == torch.float32
    assert mask.shape == (16, 16)
    assert mask.dtype == torch.int64
    assert set(mask.unique().tolist()) == {0, 1, 255}
    assert sample_id.startswith("sample_")


def test_dataset_sample_limit_keeps_manifest_order(paired_data: Path, tmp_path: Path) -> None:
    manifests = prepare_paired_manifests(
        paired_data,
        tmp_path / "manifests",
        [0.8, 0.1, 0.1],
        seed=7,
        label_map={1: 1, 2: 0},
        ignore_values={3},
    )
    transform = PairedTransform(16, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    full = SegmentationDataset(manifests["train"], transform, {1: 1, 2: 0}, [3], 255)
    limited = SegmentationDataset(manifests["train"], transform, {1: 1, 2: 0}, [3], 255, max_samples=2)
    assert len(limited) == 2
    assert [limited[index][2] for index in range(2)] == [full[index][2] for index in range(2)]

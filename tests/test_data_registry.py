import pytest

from image_segmenter.data import (
    DatasetSpec,
    available_datasets,
    dataset_status,
    get_dataset_spec,
    register_dataset,
)


def test_core_dataset_registry_exposes_builtin_providers() -> None:
    assert set(available_datasets()) >= {"oxford_pet", "paired_masks", "split_folders", "manifest"}
    assert available_datasets() == tuple(sorted(available_datasets()))
    assert get_dataset_spec("paired_masks").provider == "image-segmenter"
    assert dataset_status("paired_masks").available is True


def test_duplicate_dataset_registration_is_rejected() -> None:
    with pytest.raises(ValueError, match="already registered"):
        register_dataset(
            DatasetSpec(
                name="paired_masks",
                provider="test",
                prepare=lambda **_: {},
            )
        )


def test_unknown_dataset_suggests_close_name() -> None:
    with pytest.raises(KeyError, match="did you mean 'paired_masks'"):
        get_dataset_spec("paired_mask")

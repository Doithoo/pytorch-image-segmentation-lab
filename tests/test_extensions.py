from pathlib import Path

import pytest
import torch

from image_segmenter.data import ClassDefinition, LabelSchema, load_external_datasets
from image_segmenter.data.transforms import PairedTransform
from image_segmenter.extensions import ExtensionError, load_factory
from image_segmenter.models import create_model

FACTORY = "tests.fixtures.external_models:create_model"


def test_load_factory_imports_valid_callable_without_changing_sys_path() -> None:
    factory = load_factory(
        FACTORY,
        required_keywords=("in_channels", "num_classes", "pretrained"),
    )

    assert factory.__name__ == "create_model"


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("missing_extension_module:create_model", "could not import module"),
        ("tests.fixtures.external_models:missing", "has no attribute"),
        ("tests.fixtures.external_models:NOT_CALLABLE", "is not callable"),
        ("tests.fixtures.external_models", "module:function"),
    ],
)
def test_load_factory_reports_distinct_lookup_errors(path: str, message: str) -> None:
    with pytest.raises(ExtensionError, match=message):
        load_factory(path)


def test_load_factory_rejects_unsupported_required_parameter() -> None:
    with pytest.raises(ExtensionError, match="unsupported required parameter 'secret'"):
        load_factory(
            "tests.fixtures.external_models:requires_unknown",
            required_keywords=("in_channels", "num_classes", "pretrained"),
        )


def test_external_model_factory_builds_direct_logits_model() -> None:
    model = create_model(
        "custom_tiny",
        factory=FACTORY,
        in_channels=3,
        num_classes=3,
        pretrained=False,
        params={"width": 8},
    )

    output = model(torch.randn(2, 3, 16, 24))

    assert output.shape == (2, 3, 16, 24)
    assert model.checkpoint_metadata["factory"] == FACTORY


def test_external_model_factory_must_return_module() -> None:
    with pytest.raises(ExtensionError, match="did not return torch.nn.Module"):
        create_model(
            "invalid",
            factory="tests.fixtures.external_models:returns_object",
            in_channels=3,
            num_classes=3,
            pretrained=False,
        )


def test_external_model_output_shape_is_checked_on_first_forward() -> None:
    model = create_model(
        "wrong_shape",
        factory="tests.fixtures.external_models:wrong_shape",
        in_channels=3,
        num_classes=3,
        pretrained=False,
    )

    with pytest.raises(ExtensionError, match=r"expected \[B,3,H,W\].*got \(1, 1, 16, 16\)"):
        model(torch.randn(1, 3, 16, 16))


def _external_schema() -> LabelSchema:
    return LabelSchema(
        classes=(
            ClassDefinition(0, "background", (0, 0, 0)),
            ClassDefinition(1, "object", (255, 255, 255)),
        ),
        ignore_index=255,
    )


def test_external_dataset_factory_returns_three_validated_splits(tmp_path: Path) -> None:
    transform = PairedTransform(16, [0.0] * 3, [1.0] * 3)

    datasets = load_external_datasets(
        "tests.fixtures.external_datasets:create_dataset",
        data_dir=tmp_path / "data",
        manifest_dir=tmp_path / "manifests",
        transforms={split: transform for split in ("train", "valid", "test")},
        label_schema=_external_schema(),
        params={"length": 2},
    )

    assert set(datasets) == {"train", "valid", "test"}
    assert datasets["valid"][0][2] == "valid-0"


def test_external_dataset_factory_must_return_dataset(tmp_path: Path) -> None:
    transform = PairedTransform(16, [0.0] * 3, [1.0] * 3)
    with pytest.raises(ExtensionError, match="did not return torch.utils.data.Dataset"):
        load_external_datasets(
            "tests.fixtures.external_datasets:returns_object",
            data_dir=tmp_path,
            manifest_dir=tmp_path,
            transforms={split: transform for split in ("train", "valid", "test")},
            label_schema=_external_schema(),
            params={},
        )


def test_external_dataset_sample_contract_is_validated(tmp_path: Path) -> None:
    transform = PairedTransform(16, [0.0] * 3, [1.0] * 3)
    with pytest.raises(ExtensionError, match=r"must return \(image, target, sample_id\)"):
        load_external_datasets(
            "tests.fixtures.external_datasets:bad_item",
            data_dir=tmp_path,
            manifest_dir=tmp_path,
            transforms={split: transform for split in ("train", "valid", "test")},
            label_schema=_external_schema(),
            params={},
        )

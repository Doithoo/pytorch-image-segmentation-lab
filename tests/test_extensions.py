import pytest
import torch

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

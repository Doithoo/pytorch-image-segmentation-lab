import pytest
from torch import nn

from image_segmenter.models import (
    ModelSpec,
    available_models,
    get_model_spec,
    model_status,
    register_model,
)

CORE_MODELS = {
    "unet_small",
    "fcn_resnet50",
    "deeplabv3_resnet50",
    "deeplabv3_mobilenet_v3_large",
    "lraspp_mobilenet_v3_large",
}


def test_core_registry_exposes_maintained_models() -> None:
    assert set(available_models()) >= CORE_MODELS
    assert available_models() == tuple(sorted(available_models()))


def test_model_spec_exposes_teaching_metadata() -> None:
    spec = get_model_spec("unet_small")

    assert spec.provider == "image-segmenter"
    assert spec.supports_pretrained is False
    assert "base_channels" in spec.parameters
    assert spec.input_constraints
    assert model_status("unet_small").available is True


def test_duplicate_registration_is_rejected() -> None:
    with pytest.raises(ValueError, match="already registered"):
        register_model(
            ModelSpec(
                name="unet_small",
                provider="test",
                factory=lambda **_: nn.Identity(),
            )
        )


def test_unknown_model_suggests_close_registered_name() -> None:
    with pytest.raises(KeyError, match="did you mean 'unet_small'"):
        get_model_spec("unet_smal")

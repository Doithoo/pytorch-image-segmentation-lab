import importlib.util

import pytest
import torch

from image_segmenter.models import available_models, create_model, get_model_spec, model_status

SMP_MODELS = ("smp_unet", "smp_unetplusplus", "smp_fpn", "smp_pan")


def test_smp_models_are_discoverable_without_importing_optional_package() -> None:
    assert set(SMP_MODELS) <= set(available_models(provider="segmentation-models-pytorch"))
    for name in SMP_MODELS:
        spec = get_model_spec(name)
        assert spec.dependency == "segmentation_models_pytorch"
        assert spec.install_hint == "uv sync --extra model-zoo"
        assert "encoder_name" in spec.parameters
        assert "encoder_weights" in spec.parameters
    assert "128" in " ".join(get_model_spec("smp_pan").input_constraints)


def test_smp_model_reports_actionable_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    real_find_spec = importlib.util.find_spec

    def missing_smp(name: str, package: str | None = None):
        if name == "segmentation_models_pytorch":
            return None
        return real_find_spec(name, package)

    monkeypatch.setattr("image_segmenter.models.registry.importlib.util.find_spec", missing_smp)

    status = model_status("smp_unet")
    assert status.available is False
    assert status.reason == "missing optional dependency: segmentation_models_pytorch"
    with pytest.raises(ImportError, match="uv sync --extra model-zoo"):
        create_model("smp_unet", in_channels=3, num_classes=3, pretrained=False, params={})


@pytest.mark.model_zoo
@pytest.mark.parametrize(
    ("name", "image_size"),
    [("smp_unet", 64), ("smp_unetplusplus", 64), ("smp_fpn", 64), ("smp_pan", 128)],
)
def test_smp_model_contract_without_pretrained_weights(name: str, image_size: int) -> None:
    pytest.importorskip("segmentation_models_pytorch")
    model = create_model(
        name,
        in_channels=3,
        num_classes=3,
        pretrained=False,
        params={"encoder_name": "resnet18", "encoder_weights": None},
    ).eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, image_size, image_size))
    assert output.shape == (1, 3, image_size, image_size)
    assert model.checkpoint_metadata["provider"] == "segmentation-models-pytorch"

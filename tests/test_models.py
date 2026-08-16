import pytest
import torch

from image_segmenter.models import create_model


def test_unet_returns_same_resolution_two_class_logits() -> None:
    model = create_model("unet_small", in_channels=3, num_classes=2, base_channels=4, pretrained=False)
    output = model(torch.randn(2, 3, 32, 32))
    assert output.shape == (2, 2, 32, 32)


def test_registry_rejects_unknown_names() -> None:
    with pytest.raises(KeyError, match="unknown model"):
        create_model("missing", in_channels=3, num_classes=2, base_channels=4, pretrained=False)


def test_local_unet_rejects_pretrained_flag() -> None:
    with pytest.raises(ValueError, match="does not provide pretrained weights"):
        create_model("unet_small", in_channels=3, num_classes=2, base_channels=4, pretrained=True)


def test_unet_explains_incompatible_input_size() -> None:
    model = create_model("unet_small", in_channels=3, num_classes=2, base_channels=4, pretrained=False)
    with pytest.raises(ValueError, match="divisible by 16"):
        model(torch.randn(2, 3, 30, 30))


@pytest.mark.parametrize(
    "name",
    [
        "unet_small",
        "fcn_resnet50",
        "deeplabv3_resnet50",
        "deeplabv3_mobilenet_v3_large",
        "lraspp_mobilenet_v3_large",
    ],
)
def test_core_model_contract(name: str) -> None:
    model = create_model(name, in_channels=3, num_classes=3, pretrained=False, params={}).eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 64, 64))
    assert output.shape == (1, 3, 64, 64)

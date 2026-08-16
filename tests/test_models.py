import pytest
import torch

from image_segmenter.models import available_models, create_model


def test_unet_returns_same_resolution_two_class_logits() -> None:
    model = create_model("unet_small", in_channels=3, num_classes=2, base_channels=4, pretrained=False)
    output = model(torch.randn(2, 3, 32, 32))
    assert output.shape == (2, 2, 32, 32)


def test_registry_is_small_and_rejects_unknown_names() -> None:
    assert available_models() == ("deeplabv3_resnet50", "unet_small")
    with pytest.raises(KeyError, match="unknown model"):
        create_model("missing", in_channels=3, num_classes=2, base_channels=4, pretrained=False)


def test_local_unet_rejects_pretrained_flag() -> None:
    with pytest.raises(ValueError, match="does not provide pretrained weights"):
        create_model("unet_small", in_channels=3, num_classes=2, base_channels=4, pretrained=True)


def test_unet_explains_incompatible_input_size() -> None:
    model = create_model("unet_small", in_channels=3, num_classes=2, base_channels=4, pretrained=False)
    with pytest.raises(ValueError, match="divisible by 16"):
        model(torch.randn(2, 3, 30, 30))


def test_deeplab_returns_direct_same_resolution_logits_offline() -> None:
    model = create_model("deeplabv3_resnet50", in_channels=3, num_classes=2, base_channels=16, pretrained=False).eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 32, 32))
    assert output.shape == (1, 2, 32, 32)

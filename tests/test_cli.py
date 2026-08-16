import pytest
from torch import nn

from image_segmenter.cli import main
from image_segmenter.models import ModelSpec
from image_segmenter.models import registry as model_registry


def test_show_config_applies_json_typed_override(capsys: pytest.CaptureFixture[str]) -> None:
    result = main(["show-config", "--set", "train.epochs", "3"])
    output = capsys.readouterr().out
    assert result == 0
    assert "epochs: 3" in output


def test_expected_error_has_no_traceback(capsys: pytest.CaptureFixture[str]) -> None:
    result = main(["show-config", "--set", "data.image_size", "127"])
    error = capsys.readouterr().err
    assert result == 2
    assert error.startswith("error: invalid data.image_size")
    assert "Traceback" not in error


def test_debug_reraises_expected_error() -> None:
    with pytest.raises(ValueError, match="data.image_size"):
        main(["--debug", "show-config", "--set", "data.image_size", "127"])


def test_list_models_is_sorted_and_supports_provider_filter(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list-models", "--provider", "torchvision"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()

    names = [line.split("\t", 1)[0] for line in lines[1:]]
    assert names == sorted(names)
    assert "unet_small" not in names
    assert "fcn_resnet50" in names
    assert all("torchvision" in line for line in lines[1:])


def test_model_info_shows_metadata_and_offline_parameter_count(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject_network(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"network access attempted: {args!r} {kwargs!r}")

    monkeypatch.setattr("torch.hub.load_state_dict_from_url", reject_network)

    assert main(["model-info", "unet_small"]) == 0
    output = capsys.readouterr().out
    assert "provider: image-segmenter" in output
    assert "available: yes" in output
    assert "pretrained: no" in output
    assert "base_channels:" in output
    assert "parameter_count:" in output
    assert "Height and width must be divisible by 16." in output


def test_model_info_reports_missing_optional_dependency(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        model_registry._REGISTRY,
        "missing_optional",
        ModelSpec(
            name="missing_optional",
            provider="test",
            factory=lambda **_: nn.Identity(),
            dependency="package_that_does_not_exist_for_tests",
            install_hint="uv sync --extra model-zoo",
        ),
    )

    assert main(["model-info", "missing_optional"]) == 0
    output = capsys.readouterr().out
    assert "available: no" in output
    assert "missing optional dependency" in output
    assert "uv sync --extra model-zoo" in output
    assert "parameter_count:" not in output


def test_model_info_unknown_name_uses_registry_suggestion(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["model-info", "unet_smal"]) == 2
    assert "did you mean 'unet_small'" in capsys.readouterr().err

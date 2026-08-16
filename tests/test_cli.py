import pytest

from image_segmenter.cli import main


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

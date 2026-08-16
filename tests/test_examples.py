import re
import runpy
from pathlib import Path

import pytest
from PIL import Image

EXAMPLE_DIR = Path(__file__).parents[1] / "examples"


@pytest.mark.parametrize(
    "name",
    [
        "01_mask_and_logits.py",
        "02_paired_transform.py",
        "03_dataloader_batch.py",
        "04_minimal_training_loop.py",
    ],
)
def test_standalone_learning_examples_run_without_external_data(name: str) -> None:
    namespace = runpy.run_path(str(EXAMPLE_DIR / name), run_name=f"test_{name}")
    assert namespace["main"]([]) == 0


def test_mask_example_prints_raw_and_mapped_values(capsys) -> None:
    namespace = runpy.run_path(str(EXAMPLE_DIR / "01_mask_and_logits.py"), run_name="test_mask_values")
    assert namespace["main"]([]) == 0
    output = capsys.readouterr().out
    assert "raw values=[1, 2, 3]" in output
    assert "target values=[0, 1, 255]" in output


def test_paired_transform_example_makes_the_shared_flip_visible(capsys) -> None:
    namespace = runpy.run_path(str(EXAMPLE_DIR / "02_paired_transform.py"), run_name="test_paired_flip")
    assert namespace["main"]([]) == 0
    output = capsys.readouterr().out
    assert "before=left" in output
    assert "after=right" in output
    assert "aligned=True" in output


def test_training_example_proves_optimizer_changed_parameters(capsys) -> None:
    namespace = runpy.run_path(str(EXAMPLE_DIR / "04_minimal_training_loop.py"), run_name="test_parameter_update")
    assert namespace["main"]([]) == 0
    output = capsys.readouterr().out
    match = re.search(r"parameter_update_l1=([0-9.]+)", output)
    assert match is not None
    assert float(match.group(1)) > 0


def test_checkpoint_overlay_example_uses_saved_preprocessing(trained_run: dict[str, Path], tmp_path: Path) -> None:
    namespace = runpy.run_path(
        str(EXAMPLE_DIR / "05_checkpoint_overlay.py"),
        run_name="test_checkpoint_overlay",
    )
    output = tmp_path / "prediction.png"
    assert (
        namespace["main"](
            [
                "--checkpoint",
                str(trained_run["checkpoint"]),
                "--image",
                str(trained_run["image"]),
                "--output",
                str(output),
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    assert output.is_file()
    assert output.with_name("prediction.overlay.png").is_file()
    assert Image.open(output).mode == "L"

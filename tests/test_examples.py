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


def test_checkpoint_overlay_example_uses_saved_preprocessing(
    trained_run: dict[str, Path], tmp_path: Path
) -> None:
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

import re
import runpy
from pathlib import Path

import pytest
import torch
from PIL import Image

from image_segmenter.config import load_config
from image_segmenter.data import ClassDefinition, LabelSchema, load_external_datasets
from image_segmenter.data.transforms import PairedTransform
from image_segmenter.models import create_model

EXAMPLE_DIR = Path(__file__).parents[1] / "examples"
ROOT = EXAMPLE_DIR.parent


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


@pytest.mark.parametrize(
    ("name", "model_name"),
    [
        ("fcn_resnet50.yaml", "fcn_resnet50"),
        ("lraspp_mobilenet.yaml", "lraspp_mobilenet_v3_large"),
        ("smp_unetplusplus.yaml", "smp_unetplusplus"),
        ("custom_model_example.yaml", "custom_direct_logits"),
        ("custom_dataset_example.yaml", "unet_small"),
    ],
)
def test_extension_configs_load_with_offline_model_defaults(name: str, model_name: str) -> None:
    config = load_config(ROOT / "configs" / name)
    assert config.model.name == model_name
    assert config.model.pretrained is False
    assert config.run_name


def test_custom_model_example_runs_through_public_factory_contract() -> None:
    config = load_config(ROOT / "configs" / "custom_model_example.yaml")
    model = create_model(
        config.model.name,
        factory=config.model.factory,
        in_channels=config.model.in_channels,
        num_classes=3,
        pretrained=config.model.pretrained,
        params=config.model.params,
    )

    output = model(torch.randn(2, 3, 32, 48))

    assert output.shape == (2, 3, 32, 48)
    assert model.checkpoint_metadata["factory"] == "examples.extensions.my_segmenter:create_model"


def test_custom_dataset_example_returns_transformed_three_class_splits(tmp_path: Path) -> None:
    schema = LabelSchema(
        classes=(
            ClassDefinition(0, "background", (0, 0, 0)),
            ClassDefinition(1, "stripe", (0, 255, 0)),
            ClassDefinition(2, "block", (0, 0, 255)),
        ),
        ignore_index=255,
    )
    transform = PairedTransform(32, [0.0] * 3, [1.0] * 3)

    datasets = load_external_datasets(
        "examples.extensions.my_dataset:create_dataset",
        data_dir=tmp_path / "unused-data",
        manifest_dir=tmp_path / "unused-manifests",
        transforms={split: transform for split in ("train", "valid", "test")},
        label_schema=schema,
        params={"length": 2, "source_size": 24},
    )

    assert set(datasets) == {"train", "valid", "test"}
    for split, dataset in datasets.items():
        image, target, sample_id = dataset[0]
        assert image.shape == (3, 32, 32)
        assert target.shape == (32, 32)
        assert target.dtype == torch.long
        assert set(target.unique().tolist()) == {0, 1, 2}
        assert sample_id == f"{split}-000"

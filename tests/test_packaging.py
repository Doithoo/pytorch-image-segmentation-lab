import subprocess
from importlib.metadata import version
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 CI
    import tomli as tomllib

import image_segmenter

ROOT = Path(__file__).parents[1]


def test_package_and_distribution_versions_match() -> None:
    assert image_segmenter.__version__ == "0.1.0"
    assert version("image-segmenter") == image_segmenter.__version__


def test_project_metadata_describes_the_published_package() -> None:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        metadata = tomllib.load(handle)
    project = metadata["project"]
    assert project["readme"] == "README.md"
    assert project["requires-python"] == ">=3.10,<3.13"
    assert "Programming Language :: Python :: 3.10" in project["classifiers"]
    assert "Programming Language :: Python :: 3.12" in project["classifiers"]
    assert project["urls"]["Repository"].endswith("Doithoo/pytorch-image-segmentation-lab")
    assert any(dependency.startswith("matplotlib") for dependency in project["optional-dependencies"]["dev"])
    assert any(dependency.startswith("mypy") for dependency in project["optional-dependencies"]["dev"])
    assert any(dependency.startswith("types-PyYAML") for dependency in project["optional-dependencies"]["dev"])
    assert project["optional-dependencies"]["model-zoo"] == ["segmentation-models-pytorch>=0.5,<1"]
    assert (
        "model_zoo: tests requiring the optional model-zoo dependency"
        in metadata["tool"]["pytest"]["ini_options"]["markers"]
    )


def test_wheel_declares_typed_package_and_all_reference_configs() -> None:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        metadata = tomllib.load(handle)
    assert metadata["tool"]["setuptools"]["package-data"]["image_segmenter"] == ["py.typed"]
    config_data = metadata["tool"]["setuptools"]["data-files"]["share/image-segmenter/configs"]
    assert "configs/custom_model_example.yaml" not in config_data
    assert "configs/custom_dataset_example.yaml" not in config_data
    assert "configs/learning_minimal.yaml" in config_data
    assert {path.name for path in (ROOT / "configs").glob("*.yaml")} == {
        "learning_minimal.yaml",
        "reference_unet.yaml",
        "deeplabv3_resnet50.yaml",
        "voc2012.yaml",
        "fcn_resnet50.yaml",
        "lraspp_mobilenet.yaml",
        "smp_unetplusplus.yaml",
        "custom_model_example.yaml",
        "custom_dataset_example.yaml",
    }


def test_installed_cli_help_lists_complete_workflow() -> None:
    completed = subprocess.run(
        ["segment", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    for command in (
        "show-config",
        "list-models",
        "model-info",
        "list-datasets",
        "dataset-info",
        "inspect-data",
        "compare-runs",
        "prepare-data",
        "train",
        "evaluate",
        "predict",
    ):
        assert command in completed.stdout


def test_release_scaffolding_covers_ci_templates_and_local_check() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    for version_number in ('"3.10"', '"3.11"', '"3.12"'):
        assert version_number in workflow
    for command in (
        "ruff check",
        "ruff format --check",
        "mypy",
        "pytest",
        "python -m build",
        "twine check",
        "segment --version",
    ):
        assert command in workflow
    assert (ROOT / ".github/ISSUE_TEMPLATE/bug.yml").is_file()
    assert (ROOT / ".github/ISSUE_TEMPLATE/learning-question.yml").is_file()
    assert (ROOT / ".github/pull_request_template.md").is_file()
    assert "model-zoo:" in workflow
    assert "uv sync --locked --extra dev --extra model-zoo" in workflow
    assert "pytest -m model_zoo" in workflow
    makefile = (ROOT / "Makefile").read_text()
    assert "check: lint format typecheck test build metadata" in makefile
    assert "test-model-zoo:" in makefile
    assert "twine check dist/*" in makefile

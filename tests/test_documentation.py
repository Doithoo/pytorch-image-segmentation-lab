import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
CORE_DOCUMENTS = [
    Path("README.md"),
    Path("docs/learning-path.md"),
    Path("docs/code-tour.md"),
    Path("docs/how-it-works.md"),
    Path("docs/configuration-flow.md"),
    Path("docs/config-reference.md"),
    Path("docs/using-your-data.md"),
    Path("docs/model-zoo.md"),
    Path("docs/using-models.md"),
    Path("docs/adding-models.md"),
    Path("docs/dataset-formats.md"),
    Path("docs/adding-datasets.md"),
    Path("docs/voc2012.md"),
    Path("docs/experiments.md"),
    Path("docs/reference-run.md"),
    Path("docs/tutorial/00-basics.md"),
    Path("docs/tutorial/01-environment.md"),
    Path("docs/tutorial/02-data-and-masks.md"),
    Path("docs/tutorial/03-unet.md"),
    Path("docs/tutorial/04-training.md"),
    Path("docs/tutorial/05-evaluation-and-inference.md"),
    Path("docs/tutorial/README.md"),
    Path("docs/troubleshooting.md"),
    Path("configs/README.md"),
    Path("examples/README.md"),
    Path("src/image_segmenter/README.md"),
    Path("tests/README.md"),
    Path("CONTRIBUTING.md"),
]


def _translated(path: Path) -> Path:
    return path.with_name(f"{path.stem}.zh-CN{path.suffix}")


@pytest.mark.parametrize("english", CORE_DOCUMENTS, ids=str)
def test_core_document_has_a_chinese_companion(english: Path) -> None:
    chinese = _translated(english)
    assert (ROOT / english).is_file(), f"missing English document: {english}"
    assert (ROOT / chinese).is_file(), f"missing Chinese document: {chinese}"


def test_readme_contains_verified_beginner_commands_and_semantics() -> None:
    text = (ROOT / "README.md").read_text()
    for command in (
        "uv sync --extra dev",
        "uv run python scripts/download_data.py --data-dir data/raw",
        "uv run segment prepare-data",
        "uv run segment train --config configs/learning_minimal.yaml --dry-run",
        "uv run segment evaluate",
        "uv run segment predict",
    ):
        assert command in text
    for phrase in (
        "1 -> pet",
        "2 -> background",
        "3 -> ignored/unclassified",
        "validation pet IoU",
        "official test split",
    ):
        assert phrase in text


def test_chinese_readme_presents_a_guided_journey_and_real_results() -> None:
    text = (ROOT / "README.zh-CN.md").read_text()
    for phrase in (
        "这个项目适合谁",
        "8–12 小时",
        "dry-run OK",
        "docs/assets/reference-unet-dataset-preview.png",
        "docs/assets/reference-unet-loss-curve.png",
        "docs/assets/reference-unet-overlay.png",
        "项目结构",
    ):
        assert phrase in text


def test_public_learning_docs_do_not_contain_agent_execution_plans() -> None:
    assert not (ROOT / "docs" / "superpowers").exists()
    for document in (ROOT / "docs").rglob("*.md"):
        text = document.read_text().lower()
        assert "for agentic workers" not in text
        assert "required sub-skill" not in text


def test_extension_guides_reference_runnable_names_and_factories() -> None:
    combined = "\n".join(
        (ROOT / path).read_text()
        for path in CORE_DOCUMENTS
        if path.name
        in {"model-zoo.md", "using-models.md", "adding-models.md", "dataset-formats.md", "adding-datasets.md"}
    )
    for value in (
        "fcn_resnet50",
        "lraspp_mobilenet_v3_large",
        "smp_unetplusplus",
        "examples.extensions.my_segmenter:create_model",
        "examples.extensions.my_dataset:create_dataset",
        "uv sync --extra dev --extra model-zoo",
    ):
        assert value in combined


def test_environment_guides_cover_local_accelerators_and_kaggle_workflow() -> None:
    for path in (ROOT / "docs/tutorial/01-environment.md", ROOT / "docs/tutorial/01-environment.zh-CN.md"):
        text = path.read_text()
        for value in (
            "CUDA",
            "MPS",
            "Kaggle",
            "/kaggle/input",
            "/kaggle/working",
            "--dry-run",
            "--resume",
            "segment evaluate",
        ):
            assert value in text


def test_public_docs_avoid_artificial_completion_language() -> None:
    for document in (ROOT / "docs").rglob("*.md"):
        assert "毕业实验" not in document.read_text()


def test_relative_markdown_links_resolve() -> None:
    documents = [path for path in ROOT.rglob("*.md") if ".worktrees" not in path.parts]
    broken: list[str] = []
    for document in documents:
        text = document.read_text()
        for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", text):
            clean = target.split("#", 1)[0]
            if not clean or "://" in clean or clean.startswith("mailto:"):
                continue
            resolved = (document.parent / clean).resolve()
            if not resolved.exists():
                broken.append(f"{document.relative_to(ROOT)} -> {target}")
    assert not broken, "broken links:\n" + "\n".join(broken)

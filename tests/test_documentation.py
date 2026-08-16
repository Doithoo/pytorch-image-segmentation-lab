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
    Path("docs/experiments.md"),
    Path("docs/reference-run.md"),
    Path("docs/tutorial/00-basics.md"),
    Path("docs/tutorial/01-environment.md"),
    Path("docs/tutorial/02-data-and-masks.md"),
    Path("docs/tutorial/03-unet.md"),
    Path("docs/tutorial/04-training.md"),
    Path("docs/tutorial/05-evaluation-and-inference.md"),
    Path("configs/README.md"),
    Path("examples/README.md"),
    Path("src/image_segmenter/README.md"),
    Path("tests/README.md"),
    Path("CONTRIBUTING.md"),
]


def _translated(path: Path) -> Path:
    return path.with_name(f"{path.stem}.zh-CN{path.suffix}")


def _heading_levels(text: str) -> list[int]:
    return [len(match.group(1)) for match in re.finditer(r"^(#{1,6})\s+", text, re.MULTILINE)]


def _shell_blocks(text: str) -> list[str]:
    return [
        match.group(1).strip()
        for match in re.finditer(r"```(?:bash|console|sh)\n(.*?)```", text, re.DOTALL)
    ]


@pytest.mark.parametrize("english", CORE_DOCUMENTS, ids=str)
def test_core_document_has_structural_chinese_parity(english: Path) -> None:
    chinese = _translated(english)
    assert (ROOT / english).is_file(), f"missing English document: {english}"
    assert (ROOT / chinese).is_file(), f"missing Chinese document: {chinese}"
    english_text = (ROOT / english).read_text()
    chinese_text = (ROOT / chinese).read_text()
    assert _heading_levels(english_text) == _heading_levels(chinese_text)
    assert _shell_blocks(english_text) == _shell_blocks(chinese_text)


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

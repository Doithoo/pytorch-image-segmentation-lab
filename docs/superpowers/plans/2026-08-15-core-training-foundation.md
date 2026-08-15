# Core Training Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an installable CPU-first package that validates paired image/mask data and trains a small U-Net into reproducible best/last checkpoints.

**Architecture:** Implement one vertical training path with strict dataclass configuration, portable CSV manifests, paired torchvision transforms, a local U-Net, ignore-aware losses and metrics, and a trainer that owns checkpoint lifecycle. Keep CLI code as a presentation layer and keep evaluation formulas outside the trainer so later plans can extend evaluation and inference without changing training contracts.

**Tech Stack:** Python 3.10-3.12, PyTorch, torchvision, NumPy, Pillow, PyYAML, pytest, Ruff, setuptools, uv

---

## Plan Series And Scope

The approved design is delivered through three independently testable plans:

1. This plan: packaging, configuration, custom paired data, U-Net, loss,
   training, resume, and best/last checkpoints.
2. Evaluation plan: complete metrics, visual evidence, inference, and
   DeepLabV3 through the stable contracts created here.
3. Learning and release plan: Oxford download/splits, examples, bilingual
   documentation, reference runs, CI, and remote publication.

At the end of this plan, synthetic or custom paired data can complete:

```text
segment prepare-data -> segment train --dry-run -> segment train
```

Evaluation, prediction, Oxford download, and public release are outside this
plan and must not be stubbed into the CLI.

## File Map

Create these files, each with one primary responsibility:

```text
.gitignore                                  Generated/local file policy
LICENSE                                     MIT license
Makefile                                    Quality command aliases
pyproject.toml                              Package and tool configuration
src/image_segmenter/__init__.py             Package version
src/image_segmenter/py.typed                Typed-package marker
src/image_segmenter/cli.py                  Argument parsing and error boundary
src/image_segmenter/config.py               Typed config, loading, validation
src/image_segmenter/data/__init__.py        Public data API
src/image_segmenter/data/manifest.py        Pairing, validation, CSV manifests
src/image_segmenter/data/dataset.py         Manifest-backed Dataset
src/image_segmenter/data/transforms.py      Joint image/mask transforms
src/image_segmenter/models/__init__.py      Public model API
src/image_segmenter/models/registry.py      Model name to constructor mapping
src/image_segmenter/models/unet.py          Handwritten small U-Net
src/image_segmenter/evaluation/__init__.py  Public metric API
src/image_segmenter/evaluation/metrics.py   Ignore-aware confusion metrics
src/image_segmenter/training/__init__.py    Public training API
src/image_segmenter/training/losses.py      Cross-entropy and soft Dice
src/image_segmenter/training/checkpoint.py  Versioned checkpoint I/O
src/image_segmenter/training/trainer.py     Epoch loop and checkpoint policy
src/image_segmenter/training/train.py       Config-to-runtime assembly
configs/learning_minimal.yaml               CPU-first two-epoch configuration
tests/conftest.py                           Synthetic paired-data fixture
tests/test_packaging.py                     Package metadata checks
tests/test_config.py                        Config parsing and validation
tests/test_cli.py                           CLI behavior and errors
tests/test_manifest.py                      Pairing, validation, and splits
tests/test_dataset.py                       Tensor and transform contracts
tests/test_models.py                        U-Net and registry contracts
tests/test_losses.py                        Ignore-aware loss behavior
tests/test_metrics.py                       Known confusion-matrix results
tests/test_checkpoint.py                    Checkpoint round trips
tests/test_training.py                      Dry run, fit, best/last, resume
tests/test_end_to_end.py                     Public CLI training workflow
```

## Shared Naming Contracts

Use these names consistently in every task:

```python
BACKGROUND_CLASS = 0
FOREGROUND_CLASS = 1
IGNORE_INDEX = 255
CLASS_NAMES = ("background", "pet")
CHECKPOINT_SCHEMA_VERSION = 1
```

A dataset item is `tuple[torch.Tensor, torch.Tensor, str]`, containing image,
mask, and sample ID. A model returns logits directly, never a dictionary. An
epoch metric dictionary uses `loss`, `pet_iou`, and `valid_pixels` keys.

### Task 1: Package And Quality Foundation

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `LICENSE`
- Create: `Makefile`
- Create: `src/image_segmenter/__init__.py`
- Create: `src/image_segmenter/py.typed`
- Create: `tests/test_packaging.py`

- [ ] **Step 1: Write the failing package metadata test**

```python
# tests/test_packaging.py
from importlib.metadata import version

import image_segmenter


def test_package_and_distribution_versions_match() -> None:
    assert image_segmenter.__version__ == "0.1.0"
    assert version("image-segmenter") == image_segmenter.__version__
```

- [ ] **Step 2: Run the test to verify the package is absent**

Run: `uv run --with pytest pytest tests/test_packaging.py -v`

Expected: FAIL during import or project discovery because `pyproject.toml` and
`image_segmenter` do not exist.

- [ ] **Step 3: Add the minimal package and tool configuration**

Create `src/image_segmenter/__init__.py`:

```python
"""Beginner-oriented PyTorch semantic segmentation lab."""

__version__ = "0.1.0"
```

Create an empty `src/image_segmenter/py.typed`. Copy the sibling repository's
`../Image-Classification/LICENSE` verbatim so the copyright holder, year, and
MIT terms stay consistent. Add this project metadata:

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=77,<82"]
build-backend = "setuptools.build_meta"

[project]
name = "image-segmenter"
version = "0.1.0"
description = "Beginner-oriented reproducible PyTorch semantic segmentation"
requires-python = ">=3.10,<3.13"
license = "MIT"
license-files = ["LICENSE"]
authors = [{ name = "Yashowhoo" }]
dependencies = [
    "torch>=2.0,<3",
    "torchvision>=0.15,<1",
    "numpy>=1.24,<3",
    "pillow>=9,<12",
    "pyyaml>=6,<7",
]

[project.optional-dependencies]
dev = [
    "pytest>=7,<9",
    "ruff>=0.6,<1",
    "pre-commit>=3,<5",
    "build>=1.2,<2",
    "twine>=5,<7",
]
plot = ["matplotlib>=3.6,<4"]

[project.scripts]
segment = "image_segmenter.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
image_segmenter = ["py.typed"]

[tool.ruff]
line-length = 120
target-version = "py310"
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B", "SIM"]
ignore = ["E501"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

Create `.gitignore` with `.venv/`, `__pycache__/`, `.pytest_cache/`,
`.ruff_cache/`, `dist/`, `build/`, `*.egg-info/`, `data/raw/`, and
`artifacts/`. Create `Makefile` targets `lint`, `test`, and `build` that run
`uv run ruff check .`, `uv run pytest`, and `uv run python -m build`.

- [ ] **Step 4: Lock dependencies and run the focused test**

Run: `uv lock`

Expected: `uv.lock` is created without dependency resolution errors.

Run: `uv run pytest tests/test_packaging.py -v`

Expected: PASS.

- [ ] **Step 5: Run formatting checks for the new files**

Run: `uv run ruff check tests/test_packaging.py src/image_segmenter/__init__.py`

Expected: `All checks passed!`

- [ ] **Step 6: Commit the package foundation**

```bash
git add .gitignore LICENSE Makefile pyproject.toml uv.lock src tests/test_packaging.py
git commit -m "build(package): Establish project foundation" \
  -m "Add package metadata, quality tools, and the initial version contract."
```

### Task 2: Strict Typed Configuration

**Files:**
- Create: `src/image_segmenter/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing tests for defaults, overrides, and rejection**

```python
# tests/test_config.py
from pathlib import Path

import pytest

from image_segmenter.config import load_config


def test_defaults_match_binary_segmentation_contract() -> None:
    cfg = load_config()
    assert cfg.data.image_size == 128
    assert cfg.data.label_map == {1: 1, 2: 0}
    assert cfg.data.ignore_values == [3]
    assert cfg.loss.ignore_index == 255
    assert cfg.model.name == "unet_small"
    assert cfg.model.num_classes == 2


def test_yaml_then_dotted_override_precedence(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("train:\n  epochs: 5\n")
    cfg = load_config(path, {"train.epochs": 2, "model.base_channels": 8})
    assert cfg.train.epochs == 2
    assert cfg.model.base_channels == 8


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("train:\n  epohs: 2\n")
    with pytest.raises(ValueError, match="unknown config key: train.epohs"):
        load_config(path)


@pytest.mark.parametrize("size", [0, 127, 130])
def test_unet_image_size_must_be_positive_and_divisible_by_16(size: int) -> None:
    with pytest.raises(ValueError, match="data.image_size"):
        load_config(overrides={"data.image_size": size})
```

- [ ] **Step 2: Run tests and confirm the module is missing**

Run: `uv run pytest tests/test_config.py -v`

Expected: FAIL with `ModuleNotFoundError: image_segmenter.config`.

- [ ] **Step 3: Implement the typed configuration surface**

Define these dataclasses in `src/image_segmenter/config.py`:

```python
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    data_dir: str = "data/raw"
    manifest_dir: str = "data/manifests"
    image_size: int = 128
    normalize_mean: list[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: list[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    label_map: dict[int, int] = field(default_factory=lambda: {1: 1, 2: 0})
    ignore_values: list[int] = field(default_factory=lambda: [3])
    split_ratios: list[float] = field(default_factory=lambda: [0.8, 0.1, 0.1])
    seed: int = 666
    num_workers: int = 0
    pin_memory: bool = False
    horizontal_flip: float = 0.5
    max_train_samples: int | None = 256
    max_valid_samples: int | None = 64
    max_test_samples: int | None = 64


@dataclass
class ModelConfig:
    name: str = "unet_small"
    in_channels: int = 3
    num_classes: int = 2
    base_channels: int = 16
    pretrained: bool = False


@dataclass
class LossConfig:
    name: str = "cross_entropy"
    cross_entropy_weight: float = 1.0
    dice_weight: float = 0.0
    ignore_index: int = 255


@dataclass
class TrainConfig:
    epochs: int = 2
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "none"
    grad_clip: float = 0.0
    seed: int = 42


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    device: str = "auto"
    output_dir: str = "artifacts"
    run_name: str | None = None
    log_level: str = "info"
```

Implement `load_config(path=None, overrides=None)`, `to_dict(cfg)`, and
`dump_config(cfg, path)` with the following loading structure. Convert
`label_map` keys to integers inside `_from_dict` when `cls is DataConfig`:

```python
SECTION_TYPES = {
    "data": DataConfig,
    "model": ModelConfig,
    "loss": LossConfig,
    "train": TrainConfig,
}
TOP_LEVEL = {"device", "output_dir", "run_name", "log_level"}


def _from_dict(cls, raw: dict[str, Any], section: str):
    valid = {item.name for item in fields(cls)}
    unknown = sorted(set(raw) - valid)
    if unknown:
        raise ValueError(f"unknown config key: {section}.{unknown[0]}")
    values = dict(raw)
    if cls is DataConfig and "label_map" in values:
        values["label_map"] = {int(key): int(value) for key, value in values["label_map"].items()}
    return cls(**values)


def load_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> ExperimentConfig:
    raw: dict[str, Any] = {}
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(raw, dict):
            raise ValueError("config root must be a mapping")
    cfg = ExperimentConfig()
    for key, value in raw.items():
        if key in SECTION_TYPES and isinstance(value, dict):
            setattr(cfg, key, _from_dict(SECTION_TYPES[key], value, key))
        elif key in TOP_LEVEL:
            setattr(cfg, key, value)
        else:
            raise ValueError(f"unknown config section: {key}")
    for dotted_key, value in (overrides or {}).items():
        parts = dotted_key.split(".")
        if len(parts) == 1 and parts[0] in TOP_LEVEL:
            setattr(cfg, parts[0], value)
        elif len(parts) == 2 and parts[0] in SECTION_TYPES:
            target = getattr(cfg, parts[0])
            if parts[1] not in {item.name for item in fields(target)}:
                raise ValueError(f"unknown config key: {dotted_key}")
            if dotted_key == "data.label_map":
                value = {int(key): int(mapped) for key, mapped in value.items()}
            setattr(target, parts[1], value)
        else:
            raise ValueError(f"unknown config key: {dotted_key}")
    _validate_config(cfg)
    return cfg


def to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    return dataclasses.asdict(cfg)


def dump_config(cfg: ExperimentConfig, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(to_dict(cfg), sort_keys=False))
```

Reject unknown sections and fields before applying overrides. In addition to
the checks in the following function, validate that mean/std each contain three
numbers, std values are positive, flip probability is in `[0, 1]`, optional
sample limits are positive integers, base channels are positive, optimizer is
`adamw`, scheduler is `none` or `cosine`, both loss weights are non-negative,
at least one loss weight is positive, and `log_level` is a standard Python
logging level. Validate these core rules:

```python
def _validate_config(cfg: ExperimentConfig) -> None:
    if cfg.data.image_size <= 0 or cfg.data.image_size % 16 != 0:
        raise ValueError("invalid data.image_size; expected a positive multiple of 16")
    if cfg.model.num_classes != 2:
        raise ValueError("invalid model.num_classes; expected 2")
    if set(cfg.data.label_map.values()) - {0, 1}:
        raise ValueError("invalid data.label_map; mapped classes must be 0 or 1")
    if set(cfg.data.label_map) & set(cfg.data.ignore_values):
        raise ValueError("data.label_map and data.ignore_values must not overlap")
    if cfg.loss.ignore_index in {0, 1}:
        raise ValueError("loss.ignore_index must differ from trainable classes")
    if len(cfg.data.split_ratios) != 3 or any(value < 0 for value in cfg.data.split_ratios):
        raise ValueError("data.split_ratios must contain three non-negative values")
    if abs(sum(cfg.data.split_ratios) - 1.0) > 1e-9:
        raise ValueError("data.split_ratios must sum to 1")
    if cfg.train.epochs < 1 or cfg.train.batch_size < 1:
        raise ValueError("train.epochs and train.batch_size must be positive")
    if cfg.train.lr <= 0 or cfg.train.weight_decay < 0 or cfg.train.grad_clip < 0:
        raise ValueError("training numeric values are outside the supported range")
    if cfg.device not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("device must be auto, cpu, cuda, or mps")
```

- [ ] **Step 4: Run the configuration tests**

Run: `uv run pytest tests/test_config.py -v`

Expected: PASS for defaults, override precedence, typo rejection, and all three
invalid dimensions.

- [ ] **Step 5: Run the focused lint check**

Run: `uv run ruff check src/image_segmenter/config.py tests/test_config.py`

Expected: `All checks passed!`

- [ ] **Step 6: Commit strict configuration**

```bash
git add src/image_segmenter/config.py tests/test_config.py
git commit -m "feat(config): Add strict segmentation configuration" \
  -m "Validate labels, image dimensions, and typed command overrides early."
```

### Task 3: CLI Configuration And Error Boundary

**Files:**
- Create: `src/image_segmenter/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing CLI tests**

```python
# tests/test_cli.py
from image_segmenter.cli import main


def test_show_config_applies_json_typed_override(capsys) -> None:
    result = main(["show-config", "--set", "train.epochs", "3"])
    output = capsys.readouterr().out
    assert result == 0
    assert "epochs: 3" in output


def test_expected_error_has_no_traceback(capsys) -> None:
    result = main(["show-config", "--set", "data.image_size", "127"])
    error = capsys.readouterr().err
    assert result == 2
    assert error.startswith("error: invalid data.image_size")
    assert "Traceback" not in error
```

- [ ] **Step 2: Run the tests and verify the CLI module is absent**

Run: `uv run pytest tests/test_cli.py -v`

Expected: FAIL with `ModuleNotFoundError: image_segmenter.cli`.

- [ ] **Step 3: Implement only `show-config` and shared CLI helpers**

Implement `_parse_set()` with `json.loads` fallback, `_add_config_args()`, and
`main()`. The public behavior is:

```python
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="segment")
    parser.add_argument("--version", action="version", version=f"image-segmenter {__version__}")
    parser.add_argument("--debug", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)
    show = subparsers.add_parser("show-config")
    show.add_argument("--config")
    show.add_argument("--set", nargs=2, action="append", default=[])
    show.set_defaults(func=_show_config)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        if args.debug:
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 2
```

`_show_config` loads the YAML and typed overrides, then writes
`yaml.safe_dump(to_dict(cfg), sort_keys=False)` to stdout. Do not add unimplemented
subcommands.

- [ ] **Step 4: Run CLI and packaging tests**

Run: `uv run pytest tests/test_cli.py tests/test_packaging.py -v`

Expected: PASS.

Run: `uv run segment --version`

Expected: `image-segmenter 0.1.0`.

- [ ] **Step 5: Commit the CLI boundary**

```bash
git add src/image_segmenter/cli.py tests/test_cli.py
git commit -m "feat(cli): Add strict configuration command" \
  -m "Expose resolved YAML while keeping expected errors concise."
```

### Task 4: Synthetic Fixture And Paired Manifest Preparation

**Files:**
- Create: `tests/conftest.py`
- Create: `src/image_segmenter/data/__init__.py`
- Create: `src/image_segmenter/data/manifest.py`
- Create: `tests/test_manifest.py`

- [ ] **Step 1: Add a reusable synthetic paired-data fixture**

```python
# tests/conftest.py
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def paired_data(tmp_path: Path) -> Path:
    root = tmp_path / "paired"
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir(parents=True)
    for index in range(10):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[:, :, index % 3] = 80 + index
        mask = np.full((32, 32), 2, dtype=np.uint8)
        mask[8:24, 8:24] = 1
        mask[6:10, 6:26] = 3
        Image.fromarray(image).save(root / "images" / f"sample_{index:02d}.png")
        Image.fromarray(mask).save(root / "masks" / f"sample_{index:02d}.png")
    return root
```

- [ ] **Step 2: Write failing manifest tests**

```python
# tests/test_manifest.py
import csv
from pathlib import Path

import pytest

from image_segmenter.data.manifest import ManifestError, prepare_paired_manifests


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_prepare_pairs_and_splits_deterministically(paired_data: Path, tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    prepare_paired_manifests(
        paired_data, first, [0.8, 0.1, 0.1], seed=7, label_map={1: 1, 2: 0}, ignore_values={3}
    )
    prepare_paired_manifests(
        paired_data, second, [0.8, 0.1, 0.1], seed=7, label_map={1: 1, 2: 0}, ignore_values={3}
    )
    assert [len(_rows(first / f"{split}.csv")) for split in ("train", "valid", "test")] == [8, 1, 1]
    assert (first / "train.csv").read_text() == (second / "train.csv").read_text()


def test_missing_mask_is_reported_with_invalid_csv(paired_data: Path, tmp_path: Path) -> None:
    (paired_data / "masks" / "sample_00.png").unlink()
    output = tmp_path / "manifests"
    with pytest.raises(ManifestError, match="1 invalid sample"):
        prepare_paired_manifests(
            paired_data, output, [0.8, 0.1, 0.1], seed=7, label_map={1: 1, 2: 0}, ignore_values={3}
        )
    assert "missing mask" in (output / "invalid.csv").read_text()


def test_empty_foreground_warns_but_is_retained(paired_data: Path, tmp_path: Path) -> None:
    from PIL import Image

    Image.new("L", (32, 32), color=2).save(paired_data / "masks" / "sample_00.png")
    with pytest.warns(UserWarning, match="1 sample has no foreground"):
        paths = prepare_paired_manifests(
            paired_data,
            tmp_path / "manifests",
            [0.8, 0.1, 0.1],
            seed=7,
            label_map={1: 1, 2: 0},
            ignore_values={3},
        )
    assert sum(len(_rows(path)) for path in paths.values()) == 10
```

- [ ] **Step 3: Run tests and verify manifest code is absent**

Run: `uv run pytest tests/test_manifest.py -v`

Expected: FAIL with `ModuleNotFoundError` for the data package.

- [ ] **Step 4: Implement manifest pairing, validation, and atomic CSV output**

Define this public surface in `manifest.py`:

```python
@dataclass(frozen=True)
class ManifestRow:
    sample_id: str
    image_path: str
    mask_path: str
    split: str
    width: int
    height: int


class ManifestError(ValueError):
    pass


def prepare_paired_manifests(
    data_dir: str | Path,
    manifest_dir: str | Path,
    split_ratios: list[float],
    seed: int,
    label_map: dict[int, int],
    ignore_values: set[int],
) -> dict[str, Path]:
```

Pair supported `.jpg`, `.jpeg`, and `.png` images to `.png` masks by relative
path stem. Decode both files with Pillow, require equal sizes, require at least
one pixel whose raw value is a key in `label_map`, and reject values outside the
union of `label_map` keys and `ignore_values`. Determine foreground raw values
from entries mapped to class `1`; count masks without those values, retain them,
and emit one summarized `UserWarning` after validation. Hash exact image and mask
bytes and reject a duplicate pair before splitting. Collect all failures, write
`invalid.csv` with `sample_id,path,reason`, and raise `ManifestError` after the
full scan.

Sort valid rows by sample ID, shuffle with `random.Random(seed)`, and allocate
counts using `floor(total * train_ratio)`, `floor(total * valid_ratio)`, and the
remainder to test. Write each image and mask path relative to `manifest_dir`,
using `os.path.relpath(path, manifest_dir)` followed by `Path.as_posix()`. Write
each CSV to a sibling `.tmp` file and replace the final path only after success.
Return a mapping from split name to path.

Export `ManifestError`, `ManifestRow`, and `prepare_paired_manifests` from
`data/__init__.py`.

- [ ] **Step 5: Run manifest tests and lint**

Run: `uv run pytest tests/test_manifest.py -v`

Expected: PASS with exact 8/1/1 deterministic split sizes and aggregated invalid
data reporting.

Run: `uv run ruff check src/image_segmenter/data tests/conftest.py tests/test_manifest.py`

Expected: `All checks passed!`

- [ ] **Step 6: Commit paired manifest preparation**

```bash
git add src/image_segmenter/data tests/conftest.py tests/test_manifest.py
git commit -m "feat(data): Prepare validated paired manifests" \
  -m "Pair masks deterministically and report all invalid samples together."
```

### Task 5: Paired Transforms And Manifest Dataset

**Files:**
- Create: `src/image_segmenter/data/transforms.py`
- Create: `src/image_segmenter/data/dataset.py`
- Modify: `src/image_segmenter/data/__init__.py`
- Create: `tests/test_dataset.py`

- [ ] **Step 1: Write failing transform and dataset tests**

```python
# tests/test_dataset.py
from pathlib import Path

import torch

from image_segmenter.data.dataset import SegmentationDataset, map_mask
from image_segmenter.data.manifest import prepare_paired_manifests
from image_segmenter.data.transforms import PairedTransform


def test_map_mask_preserves_classes_and_marks_ignore() -> None:
    raw = torch.tensor([[1, 2, 3]], dtype=torch.int64)
    mapped = map_mask(raw, {1: 1, 2: 0}, [3], ignore_index=255)
    assert mapped.tolist() == [[1, 0, 255]]


def test_dataset_returns_aligned_tensor_contract(paired_data: Path, tmp_path: Path) -> None:
    manifests = prepare_paired_manifests(
        paired_data,
        tmp_path / "manifests",
        [0.8, 0.1, 0.1],
        seed=7,
        label_map={1: 1, 2: 0},
        ignore_values={3},
    )
    transform = PairedTransform(16, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], horizontal_flip=1.0)
    dataset = SegmentationDataset(manifests["train"], transform, {1: 1, 2: 0}, [3], 255)
    image, mask, sample_id = dataset[0]
    assert image.shape == (3, 16, 16)
    assert image.dtype == torch.float32
    assert mask.shape == (16, 16)
    assert mask.dtype == torch.int64
    assert set(mask.unique().tolist()) == {0, 1, 255}
    assert sample_id.startswith("sample_")
```

- [ ] **Step 2: Run tests and verify imports fail**

Run: `uv run pytest tests/test_dataset.py -v`

Expected: FAIL because `dataset.py` and `transforms.py` do not exist.

- [ ] **Step 3: Implement joint resize and flip**

Create `PairedTransform` with this behavior:

```python
class PairedTransform:
    def __init__(
        self,
        image_size: int,
        mean: list[float],
        std: list[float],
        horizontal_flip: float = 0.0,
    ) -> None:
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.horizontal_flip = horizontal_flip

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.horizontal_flip:
            image = F.hflip(image)
            mask = F.hflip(mask)
        size = [self.image_size, self.image_size]
        image = F.resize(image, size, interpolation=InterpolationMode.BILINEAR, antialias=True)
        mask = F.resize(mask, size, interpolation=InterpolationMode.NEAREST)
        image_tensor = F.normalize(F.to_tensor(image), self.mean, self.std)
        mask_tensor = F.pil_to_tensor(mask).squeeze(0).to(torch.int64)
        return image_tensor, mask_tensor
```

Use `random` from the standard library and torchvision functional transforms.
Do not apply photometric operations to masks.

- [ ] **Step 4: Implement label mapping and the manifest-backed Dataset**

`map_mask` must initialize every output pixel to `ignore_index`, assign mapped
classes by equality masks, assign configured ignored raw values explicitly, and
raise if any input value belongs to neither collection:

```python
def map_mask(raw: torch.Tensor, label_map: dict[int, int], ignore_values: list[int], ignore_index: int) -> torch.Tensor:
    known = set(label_map) | set(ignore_values)
    found = set(raw.unique().tolist())
    unknown = sorted(found - known)
    if unknown:
        raise ValueError(f"mask contains unmapped values: {unknown}")
    mapped = torch.full_like(raw, ignore_index, dtype=torch.int64)
    for raw_value, class_index in label_map.items():
        mapped[raw == raw_value] = class_index
    return mapped
```

`SegmentationDataset` reads manifest rows once in `__init__`, resolves each
relative image/mask path against `manifest_path.parent`, applies the paired
transform, maps the mask, and returns `(image, mask, sample_id)`. Add an optional
`max_samples`; when set, retain the first N manifest rows because selection was
already deterministically shuffled during preparation.

Export `SegmentationDataset`, `PairedTransform`, and `map_mask` from the data
package.

- [ ] **Step 5: Run dataset and manifest tests**

Run: `uv run pytest tests/test_dataset.py tests/test_manifest.py -v`

Expected: PASS.

Run: `uv run ruff check src/image_segmenter/data tests/test_dataset.py`

Expected: `All checks passed!`

- [ ] **Step 6: Commit the tensor data path**

```bash
git add src/image_segmenter/data tests/test_dataset.py
git commit -m "feat(data): Load aligned image and mask tensors" \
  -m "Keep geometry paired and map raw labels to the training contract."
```

### Task 6: Handwritten U-Net And Model Registry

**Files:**
- Create: `src/image_segmenter/models/__init__.py`
- Create: `src/image_segmenter/models/unet.py`
- Create: `src/image_segmenter/models/registry.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write failing model contract tests**

```python
# tests/test_models.py
import pytest
import torch

from image_segmenter.models import available_models, create_model


def test_unet_returns_same_resolution_two_class_logits() -> None:
    model = create_model("unet_small", in_channels=3, num_classes=2, base_channels=4, pretrained=False)
    output = model(torch.randn(2, 3, 32, 32))
    assert output.shape == (2, 2, 32, 32)


def test_registry_is_small_and_rejects_unknown_names() -> None:
    assert available_models() == ("unet_small",)
    with pytest.raises(KeyError, match="unknown model"):
        create_model("missing", in_channels=3, num_classes=2, base_channels=4, pretrained=False)
```

- [ ] **Step 2: Run tests and verify model imports fail**

Run: `uv run pytest tests/test_models.py -v`

Expected: FAIL because the models package does not exist.

- [ ] **Step 3: Implement the complete small U-Net**

Create these focused modules in `unet.py`:

```python
class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.block = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        value = self.up(value)
        if value.shape[-2:] != skip.shape[-2:]:
            raise ValueError("U-Net skip shapes do not match; use dimensions divisible by 16")
        return self.block(torch.cat([value, skip], dim=1))


class SmallUNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 2, base_channels: int = 16) -> None:
        super().__init__()
        widths = [base_channels * factor for factor in (1, 2, 4, 8)]
        self.encoders = nn.ModuleList(
            [
                ConvBlock(in_channels, widths[0]),
                ConvBlock(widths[0], widths[1]),
                ConvBlock(widths[1], widths[2]),
                ConvBlock(widths[2], widths[3]),
            ]
        )
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(widths[3], widths[3] * 2)
        self.decoders = nn.ModuleList(
            [
                UpBlock(widths[3] * 2, widths[3], widths[3]),
                UpBlock(widths[3], widths[2], widths[2]),
                UpBlock(widths[2], widths[1], widths[1]),
                UpBlock(widths[1], widths[0], widths[0]),
            ]
        )
        self.head = nn.Conv2d(widths[0], num_classes, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        value = image
        for encoder in self.encoders:
            value = encoder(value)
            skips.append(value)
            value = self.pool(value)
        value = self.bottleneck(value)
        for decoder, skip in zip(self.decoders, reversed(skips), strict=True):
            value = decoder(value, skip)
        return self.head(value)
```

- [ ] **Step 4: Implement the one-model registry**

`available_models()` returns a sorted tuple. `create_model()` accepts the exact
keyword signature used by the test, rejects `pretrained=True` for `unet_small`
with a clear `ValueError`, and constructs `SmallUNet`. Export the three public
symbols from `models/__init__.py`.

- [ ] **Step 5: Run model tests and lint**

Run: `uv run pytest tests/test_models.py -v`

Expected: PASS.

Run: `uv run ruff check src/image_segmenter/models tests/test_models.py`

Expected: `All checks passed!`

- [ ] **Step 6: Commit the teaching model**

```bash
git add src/image_segmenter/models tests/test_models.py
git commit -m "feat(models): Add readable small U-Net" \
  -m "Expose a same-resolution binary segmentation model through a registry."
```

### Task 7: Ignore-Aware Loss And Core Metrics

**Files:**
- Create: `src/image_segmenter/training/__init__.py`
- Create: `src/image_segmenter/training/losses.py`
- Create: `src/image_segmenter/evaluation/__init__.py`
- Create: `src/image_segmenter/evaluation/metrics.py`
- Create: `tests/test_losses.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write failing loss tests with hand-controlled logits**

```python
# tests/test_losses.py
import torch

from image_segmenter.training.losses import SegmentationLoss, foreground_soft_dice_loss


def test_ignored_logit_changes_do_not_change_dice() -> None:
    target = torch.tensor([[[1, 255]]])
    first = torch.tensor([[[[0.0, -5.0]], [[5.0, 5.0]]]])
    second = first.clone()
    second[:, :, 0, 1] = torch.tensor([100.0, -100.0])
    assert torch.allclose(
        foreground_soft_dice_loss(first, target, 255),
        foreground_soft_dice_loss(second, target, 255),
    )


def test_empty_foreground_uses_cross_entropy_only() -> None:
    criterion = SegmentationLoss(1.0, 1.0, ignore_index=255)
    logits = torch.zeros(1, 2, 2, 2, requires_grad=True)
    target = torch.zeros(1, 2, 2, dtype=torch.int64)
    total, components = criterion(logits, target)
    assert components["dice_active"] == 0.0
    assert torch.allclose(total, components["cross_entropy"])
```

- [ ] **Step 2: Write failing metric tests from a known confusion matrix**

```python
# tests/test_metrics.py
import pytest
import torch

from image_segmenter.evaluation.metrics import SegmentationMetrics


def test_metrics_ignore_255_and_report_foreground_iou() -> None:
    metrics = SegmentationMetrics(num_classes=2, ignore_index=255)
    prediction = torch.tensor([[[0, 1], [1, 0]]])
    target = torch.tensor([[[0, 1], [0, 255]]])
    metrics.update(prediction, target)
    result = metrics.compute()
    assert result["valid_pixels"] == 3
    assert result["pet_iou"] == pytest.approx(0.5)
    assert result["background_iou"] == pytest.approx(0.5)
    assert result["mean_iou"] == pytest.approx(0.5)
```

- [ ] **Step 3: Run tests and verify loss and metric modules are absent**

Run: `uv run pytest tests/test_losses.py tests/test_metrics.py -v`

Expected: FAIL during module import.

- [ ] **Step 4: Implement the loss contract**

Use `F.cross_entropy(logits, target, ignore_index=ignore_index)`. For Dice,
softmax logits, select channel 1, select only valid pixels, and compute
`1 - (2 * intersection + 1e-6) / (prediction_sum + target_sum + 1e-6)`.

`SegmentationLoss.forward()` returns `(total, components)`. `components` maps
`cross_entropy`, `dice`, and `dice_active` to detached scalar tensors. If no
valid foreground exists, use a graph-connected zero Dice value
`logits.sum() * 0.0`; otherwise apply the configured Dice weight. Reject a batch
with no valid pixels.

- [ ] **Step 5: Implement streaming confusion metrics**

`SegmentationMetrics` stores an `int64 [2, 2]` CPU confusion matrix. `update`
argmaxes four-dimensional logits or accepts three-dimensional class indices,
filters ignored pixels, and uses:

```python
encoded = target_valid * self.num_classes + prediction_valid
counts = torch.bincount(encoded, minlength=self.num_classes**2)
self.confusion += counts.reshape(self.num_classes, self.num_classes).cpu()
```

`compute()` derives per-class intersection from the diagonal and union as
row sum plus column sum minus diagonal. Return `background_iou`, `pet_iou`,
`mean_iou`, `pet_dice`, `macro_dice`, `pixel_accuracy`, and `valid_pixels`.
Use `NaN` for a class with zero union and exclude it from macro means.

- [ ] **Step 6: Run focused tests and lint**

Run: `uv run pytest tests/test_losses.py tests/test_metrics.py -v`

Expected: PASS.

Run: `uv run ruff check src/image_segmenter/training/losses.py src/image_segmenter/evaluation tests/test_losses.py tests/test_metrics.py`

Expected: `All checks passed!`

- [ ] **Step 7: Commit loss and metrics**

```bash
git add src/image_segmenter/training src/image_segmenter/evaluation tests/test_losses.py tests/test_metrics.py
git commit -m "feat(training): Add ignore-aware loss and metrics" \
  -m "Exclude uncertain pixels and prioritize foreground segmentation quality."
```

### Task 8: Versioned Checkpoint Round Trip

**Files:**
- Create: `src/image_segmenter/training/checkpoint.py`
- Modify: `src/image_segmenter/training/__init__.py`
- Create: `tests/test_checkpoint.py`

- [ ] **Step 1: Write failing checkpoint tests**

```python
# tests/test_checkpoint.py
from pathlib import Path

import pytest
import torch

from image_segmenter.config import load_config
from image_segmenter.models import create_model
from image_segmenter.training.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_restores_model_optimizer_and_metadata(tmp_path: Path) -> None:
    cfg = load_config(overrides={"model.base_channels": 4, "data.image_size": 32})
    model = create_model("unet_small", in_channels=3, num_classes=2, base_channels=4, pretrained=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    path = tmp_path / "last.pt"
    save_checkpoint(path, model, optimizer, None, epoch=3, best_metric=0.6, cfg=cfg, manifest_checksums={"train": "abc"})
    loaded = load_checkpoint(path)
    assert loaded["schema_version"] == 1
    assert loaded["epoch"] == 3
    assert loaded["best_metric"] == 0.6
    assert loaded["config"]["data"]["label_map"] == {1: 1, 2: 0}


def test_unknown_checkpoint_schema_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "future.pt"
    torch.save({"schema_version": 99}, path)
    with pytest.raises(ValueError, match="checkpoint schema 99 is not supported"):
        load_checkpoint(path)
```

- [ ] **Step 2: Run tests and confirm checkpoint module is absent**

Run: `uv run pytest tests/test_checkpoint.py -v`

Expected: FAIL during import.

- [ ] **Step 3: Implement schema-one save and load**

Set `CHECKPOINT_SCHEMA_VERSION = 1`. `save_checkpoint()` creates the parent
directory and writes a dictionary containing:

```python
payload = {
    "schema_version": CHECKPOINT_SCHEMA_VERSION,
    "epoch": epoch,
    "best_metric": best_metric,
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
    "config": to_dict(cfg),
    "class_names": ["background", "pet"],
    "label_map": cfg.data.label_map,
    "ignore_index": cfg.loss.ignore_index,
    "manifest_checksums": manifest_checksums,
    "rng_state": {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    },
    "package_version": __version__,
}
```

Write to `<name>.tmp` and atomically replace the final checkpoint. Add the CUDA
RNG state only when CUDA is available. `load_checkpoint()` verifies the path,
loads on CPU with `torch.load(path, map_location="cpu", weights_only=False)`,
checks the schema, and checks the required keys before returning the dictionary.
Add `restore_training_state()` to load model, optimizer,
scheduler, and RNG state and return `(epoch, best_metric)`.

- [ ] **Step 4: Run checkpoint and config tests**

Run: `uv run pytest tests/test_checkpoint.py tests/test_config.py -v`

Expected: PASS.

- [ ] **Step 5: Commit checkpoint persistence**

```bash
git add src/image_segmenter/training tests/test_checkpoint.py
git commit -m "feat(training): Save versioned training checkpoints" \
  -m "Persist reproducibility metadata and reject incompatible schemas."
```

### Task 9: CPU Trainer, Dry Run, And Resume

**Files:**
- Create: `src/image_segmenter/training/trainer.py`
- Create: `src/image_segmenter/training/train.py`
- Modify: `src/image_segmenter/training/__init__.py`
- Create: `tests/test_training.py`

- [ ] **Step 1: Write failing trainer lifecycle tests**

```python
# tests/test_training.py
from pathlib import Path

from image_segmenter.config import load_config
from image_segmenter.data.manifest import prepare_paired_manifests
from image_segmenter.training.train import train_from_config


def _config(paired_data: Path, tmp_path: Path):
    manifest_dir = tmp_path / "manifests"
    prepare_paired_manifests(
        paired_data, manifest_dir, [0.8, 0.1, 0.1], 7, label_map={1: 1, 2: 0}, ignore_values={3}
    )
    return load_config(
        overrides={
            "data.data_dir": str(paired_data),
            "data.manifest_dir": str(manifest_dir),
            "data.image_size": 32,
            "data.max_train_samples": 4,
            "data.max_valid_samples": 1,
            "model.base_channels": 4,
            "train.batch_size": 2,
            "train.epochs": 1,
            "device": "cpu",
            "output_dir": str(tmp_path / "artifacts"),
            "run_name": "tiny-run",
        }
    )


def test_dry_run_does_not_create_normal_artifacts(paired_data: Path, tmp_path: Path, capsys) -> None:
    cfg = _config(paired_data, tmp_path)
    result = train_from_config(cfg, dry_run=True)
    assert result is None
    assert "dry-run OK" in capsys.readouterr().out
    assert not (Path(cfg.output_dir) / "tiny-run").exists()


def test_fit_writes_best_last_metrics_and_resumes(paired_data: Path, tmp_path: Path) -> None:
    cfg = _config(paired_data, tmp_path)
    run_dir = train_from_config(cfg)
    assert run_dir == Path(cfg.output_dir) / "tiny-run"
    assert (run_dir / "best.pt").is_file()
    assert (run_dir / "last.pt").is_file()
    assert (run_dir / "metrics.csv").read_text().count("\n") == 2
    cfg.train.epochs = 2
    train_from_config(cfg, resume=run_dir / "last.pt")
    assert (run_dir / "metrics.csv").read_text().count("\n") == 3
```

- [ ] **Step 2: Run tests and confirm the training assembler is absent**

Run: `uv run pytest tests/test_training.py -v`

Expected: FAIL importing `image_segmenter.training.train`.

- [ ] **Step 3: Implement device, seed, loader, and optimizer assembly**

In `train.py`, implement `resolve_device()` so explicit unavailable devices
raise `ValueError`, while `auto` selects CUDA, MPS, then CPU. Implement
`seed_everything()` for Python, NumPy, and torch. Build train and validation
datasets from `train.csv` and `valid.csv`; use configured sample limits, shuffle
only training, and seed the DataLoader generator.

Build the model through `create_model`, `SegmentationLoss` from loss weights,
AdamW, and `CosineAnnealingLR` only when `scheduler == "cosine"`. Reject other
optimizer and scheduler names. Calculate SHA-256 checksums of both manifest
files for checkpoint metadata. When `run_name` is `None`, resolve it once as
`<model-name>-YYYYMMDD-HHMMSS` in local time and use that value in the dumped
effective configuration and artifact path.

- [ ] **Step 4: Implement the epoch loop and dry run**

In `trainer.py`, `_run_epoch(loader, training)` must:

```python
if training:
    self.model.train()
else:
    self.model.eval()
for images, masks, _sample_ids in loader:
    images = images.to(self.device)
    masks = masks.to(self.device)
    if training:
        self.optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(training):
        logits = self.model(images)
        loss, _components = self.criterion(logits, masks)
    if training:
        loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
    metrics.update(logits.detach(), masks)
    loss_sum += loss.detach().item() * images.shape[0]
    sample_count += images.shape[0]
```

Return metric results plus mean loss. The dry run uses the first training batch,
performs exactly one update, prints device, tensor shapes, unique mask values,
valid-pixel ratio, loss, and `dry-run OK`, then returns without creating the run
directory.

- [ ] **Step 5: Implement fit, best/last policy, and resume**

Create the run directory only for a normal fit. Refuse to start a non-resume run
when the directory already contains checkpoints. Dump the effective config.
Create `metrics.csv` with this fixed header:

```text
epoch,train_loss,train_pet_iou,valid_loss,valid_pet_iou,lr
```

After every epoch, step the scheduler, append and flush one metrics row, save
`last.pt`, and save `best.pt` only when validation `pet_iou` is strictly greater
than the previous best. Resume validates model name, class mapping, ignore
index, and manifest checksums before restoring state. Continue from
`checkpoint_epoch + 1` and append without rewriting the CSV header.

- [ ] **Step 6: Run trainer lifecycle tests**

Run: `uv run pytest tests/test_training.py tests/test_checkpoint.py -v`

Expected: PASS; the resume assertion shows two data rows after the second call.

Run: `uv run ruff check src/image_segmenter/training tests/test_training.py`

Expected: `All checks passed!`

- [ ] **Step 7: Commit the CPU training lifecycle**

```bash
git add src/image_segmenter/training tests/test_training.py
git commit -m "feat(training): Train and resume U-Net runs" \
  -m "Add CPU dry runs and foreground-IoU best checkpoint selection."
```

### Task 10: Public Prepare And Train Workflow

**Files:**
- Modify: `src/image_segmenter/cli.py`
- Create: `configs/learning_minimal.yaml`
- Create: `tests/test_end_to_end.py`

- [ ] **Step 1: Write a failing public CLI end-to-end test**

```python
# tests/test_end_to_end.py
from pathlib import Path

from image_segmenter.cli import main


def test_prepare_dry_run_and_train_through_cli(paired_data: Path, tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    artifacts = tmp_path / "artifacts"
    assert main(
        [
            "prepare-data",
            "--data-dir", str(paired_data),
            "--manifest-dir", str(manifests),
            "--set", "data.split_ratios", "[0.8, 0.1, 0.1]",
        ]
    ) == 0
    common = [
        "--set", "data.manifest_dir", str(manifests),
        "--set", "data.image_size", "32",
        "--set", "data.max_train_samples", "4",
        "--set", "data.max_valid_samples", "1",
        "--set", "model.base_channels", "4",
        "--set", "train.batch_size", "2",
        "--set", "train.epochs", "1",
        "--set", "output_dir", str(artifacts),
        "--set", "run_name", "cli-run",
        "--device", "cpu",
    ]
    assert main(["train", "--dry-run", *common]) == 0
    assert main(["train", *common]) == 0
    assert (artifacts / "cli-run" / "best.pt").is_file()
```

- [ ] **Step 2: Run the test and verify subcommands are unavailable**

Run: `uv run pytest tests/test_end_to_end.py -v`

Expected: FAIL because argparse rejects `prepare-data` and `train`.

- [ ] **Step 3: Wire `prepare-data` without domain logic in the CLI**

Add config arguments to the command, plus dedicated `--data-dir` and
`--manifest-dir`. Resolve the config and pass `label_map` and
`set(ignore_values)` to
`prepare_paired_manifests`. Print the three output manifest paths. Add
`ManifestError` to the expected error boundary.

Do not parse images, masks, or CSV inside `cli.py`.

- [ ] **Step 4: Wire `train`, `--dry-run`, and `--resume`**

Add shared config arguments, `--device`, `--output-dir`, `--dry-run`, and
`--resume`. Dedicated `--device` and `--output-dir` values override config after
YAML and `--set`. Call `train_from_config`; print the run directory only when a
normal training run returns one.

Create the exact learning configuration:

```yaml
# configs/learning_minimal.yaml
data:
  data_dir: data/raw
  manifest_dir: data/manifests
  image_size: 128
  label_map: {1: 1, 2: 0}
  ignore_values: [3]
  split_ratios: [0.8, 0.1, 0.1]
  num_workers: 0
  pin_memory: false
  horizontal_flip: 0.5
  max_train_samples: 256
  max_valid_samples: 64
  max_test_samples: 64
model:
  name: unet_small
  in_channels: 3
  num_classes: 2
  base_channels: 16
  pretrained: false
loss:
  name: cross_entropy
  cross_entropy_weight: 1.0
  dice_weight: 0.0
  ignore_index: 255
train:
  epochs: 2
  batch_size: 8
  lr: 0.001
  weight_decay: 0.0001
  optimizer: adamw
  scheduler: none
  grad_clip: 0.0
  seed: 42
device: auto
output_dir: artifacts
run_name: null
log_level: info
```

- [ ] **Step 5: Run the end-to-end and full test suites**

Run: `uv run pytest tests/test_end_to_end.py -v`

Expected: PASS and both dry-run and one-epoch training complete on CPU.

Run: `uv run pytest`

Expected: all tests PASS.

Run: `uv run ruff check .`

Expected: `All checks passed!`

Run: `uv run python -m build`

Expected: wheel and source distribution are built successfully.

- [ ] **Step 6: Verify the staged diff contains no generated artifacts**

Run: `git status --short`

Expected: source, tests, config, and lockfile changes only; no `data/raw/`,
`artifacts/`, `.venv/`, or build output is staged.

- [ ] **Step 7: Commit the first vertical workflow**

```bash
git add src/image_segmenter/cli.py configs/learning_minimal.yaml tests/test_end_to_end.py
git commit -m "feat(cli): Complete the core training workflow" \
  -m "Expose paired data preparation, dry runs, training, and resume."
```

## Phase Completion Check

Run these commands from a clean checkout after all ten task commits:

```bash
uv sync --extra dev
uv run ruff check .
uv run pytest
uv run python -m build
uv run segment --version
uv run segment show-config --config configs/learning_minimal.yaml
```

Expected outcomes:

- lint reports no findings;
- all tests pass without network access;
- wheel and source distribution build;
- version is `image-segmenter 0.1.0`;
- resolved YAML shows a 128 x 128, two-class, CPU-compatible U-Net run;
- `git status --short` is empty after the final commit.

Do not claim the full approved design is complete after this plan. The next plan
must add complete evaluation, inference, visual artifacts, per-image metrics,
and DeepLabV3 before the learning and release plan begins.

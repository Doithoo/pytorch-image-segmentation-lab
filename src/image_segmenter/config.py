"""Typed experiment configuration with strict YAML and override parsing."""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    """Dataset paths, label semantics, and preprocessing."""

    name: str = "oxford_pet"
    factory: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
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
    """Segmentation model construction."""

    name: str = "unet_small"
    factory: str | None = None
    in_channels: int = 3
    expected_num_classes: int | None = 2
    pretrained: bool = False
    params: dict[str, Any] = field(default_factory=lambda: {"base_channels": 16})

    @property
    def num_classes(self) -> int:
        """Expose the v1 expectation until model creation uses LabelSchema."""
        return self.expected_num_classes if self.expected_num_classes is not None else 2

    @property
    def base_channels(self) -> int:
        """Expose the legacy U-Net parameter to compatibility call sites."""
        value = self.params.get("base_channels", 16)
        return value if isinstance(value, int) and not isinstance(value, bool) else 0


@dataclass
class LossConfig:
    """Loss composition and ignored target value."""

    name: str = "cross_entropy"
    cross_entropy_weight: float = 1.0
    dice_weight: float = 0.0
    ignore_index: int = 255


@dataclass
class TrainConfig:
    """Optimizer and training-loop settings."""

    epochs: int = 2
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "none"
    grad_clip: float = 0.0
    seed: int = 42
    amp: bool = False
    best_metric: str = "class_iou:pet"


@dataclass
class ExperimentConfig:
    """Complete resolved experiment configuration."""

    config_version: int = 2
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    device: str = "auto"
    output_dir: str = "artifacts"
    run_name: str | None = None
    log_level: str = "info"


SECTION_TYPES = {
    "data": DataConfig,
    "model": ModelConfig,
    "loss": LossConfig,
    "train": TrainConfig,
}
TOP_LEVEL = {"device", "output_dir", "run_name", "log_level"}
SUPPORTED_MODEL_NAMES = {
    "unet_small",
    "fcn_resnet50",
    "deeplabv3_resnet50",
    "deeplabv3_mobilenet_v3_large",
    "lraspp_mobilenet_v3_large",
    "smp_unet",
    "smp_unetplusplus",
    "smp_fpn",
    "smp_pan",
}


def _from_dict(cls: type[Any], raw: dict[str, Any], section: str) -> Any:
    valid = {item.name for item in fields(cls)}
    unknown = sorted(set(raw) - valid)
    if unknown:
        raise ValueError(f"unknown config key: {section}.{unknown[0]}")
    values = dict(raw)
    if cls is DataConfig and "label_map" in values:
        label_map = values["label_map"]
        if not isinstance(label_map, dict):
            raise ValueError("data.label_map must be a mapping")
        values["label_map"] = {int(key): int(value) for key, value in label_map.items()}
    return cls(**values)


def migrate_config_v1(raw: dict[str, Any]) -> dict[str, Any]:
    """Return an equivalent v2 mapping without mutating a v1 input."""
    migrated = copy.deepcopy(raw)
    migrated.pop("config_version", None)
    model = migrated.get("model")
    if model is not None:
        if not isinstance(model, dict):
            raise ValueError("config section model must be a mapping")
        params = model.get("params", {})
        if not isinstance(params, dict):
            raise ValueError("model.params must be a mapping")
        params = dict(params)
        if "base_channels" in model:
            params["base_channels"] = model.pop("base_channels")
        if params:
            model["params"] = params
        if "num_classes" in model:
            model["expected_num_classes"] = model.pop("num_classes")
    migrated["config_version"] = 2
    return migrated


def load_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> ExperimentConfig:
    """Load defaults, then YAML, then dotted overrides."""
    raw: dict[str, Any] = {}
    if path is not None:
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"config file not found: {config_path}")
        loaded = yaml.safe_load(config_path.read_text())
        raw = loaded or {}
        if not isinstance(raw, dict):
            raise ValueError("config root must be a mapping")

    return config_from_dict(raw, overrides=overrides)


def config_from_dict(
    raw: dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> ExperimentConfig:
    """Restore and validate a configuration from built-in containers."""
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")
    version = raw.get("config_version", 1)
    if not isinstance(version, int) or isinstance(version, bool) or version not in (1, 2):
        raise ValueError("config_version must be 1 or 2")
    resolved = migrate_config_v1(raw) if version == 1 else copy.deepcopy(raw)
    cfg = ExperimentConfig()
    for key, value in resolved.items():
        if key in SECTION_TYPES:
            if not isinstance(value, dict):
                raise ValueError(f"config section {key} must be a mapping")
            setattr(cfg, key, _from_dict(SECTION_TYPES[key], value, key))
        elif key in TOP_LEVEL:
            setattr(cfg, key, value)
        elif key == "config_version":
            cfg.config_version = value
        else:
            raise ValueError(f"unknown config section: {key}")

    for dotted_key, value in (overrides or {}).items():
        parts = dotted_key.split(".")
        if len(parts) == 1 and parts[0] in TOP_LEVEL:
            setattr(cfg, parts[0], value)
        elif len(parts) == 2 and parts[0] in SECTION_TYPES:
            target = getattr(cfg, parts[0])
            if dotted_key == "model.base_channels":
                _positive_int(dotted_key, value)
                target.params["base_channels"] = value
                continue
            if dotted_key == "model.num_classes":
                _positive_int(dotted_key, value)
                target.expected_num_classes = value
                continue
            if parts[1] not in {item.name for item in fields(target)}:
                raise ValueError(f"unknown config key: {dotted_key}")
            if dotted_key == "data.label_map":
                if not isinstance(value, dict):
                    raise ValueError("data.label_map must be a mapping")
                value = {int(key): int(mapped) for key, mapped in value.items()}
            setattr(target, parts[1], value)
        elif len(parts) == 3 and parts[0] in SECTION_TYPES and parts[1] == "params" and parts[2]:
            target = getattr(cfg, parts[0])
            target.params[parts[2]] = value
        else:
            raise ValueError(f"unknown config key: {dotted_key}")

    _validate_config(cfg)
    return cfg


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _positive_optional_int(key: str, value: object) -> None:
    if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
        raise ValueError(f"{key} must be null or a positive integer")


def _positive_int(key: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")


def _validate_seed(key: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < 2**32:
        raise ValueError(f"{key} must be an integer between 0 and {2**32 - 1}")


def _nonempty_string(key: str, value: object) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")


def _validate_config(cfg: ExperimentConfig) -> None:
    if cfg.config_version != 2:
        raise ValueError("resolved config_version must be 2")
    _nonempty_string("data.name", cfg.data.name)
    if cfg.data.factory is not None:
        _nonempty_string("data.factory", cfg.data.factory)
    if not isinstance(cfg.data.params, dict) or any(not isinstance(key, str) for key in cfg.data.params):
        raise ValueError("data.params must be a mapping with string keys")
    _nonempty_string("data.data_dir", cfg.data.data_dir)
    _nonempty_string("data.manifest_dir", cfg.data.manifest_dir)
    if not isinstance(cfg.data.image_size, int) or isinstance(cfg.data.image_size, bool):
        raise ValueError("invalid data.image_size; expected a positive multiple of 16")
    if cfg.data.image_size <= 0 or cfg.data.image_size % 16 != 0:
        raise ValueError("invalid data.image_size; expected a positive multiple of 16")
    if cfg.model.expected_num_classes is not None:
        _positive_int("model.expected_num_classes", cfg.model.expected_num_classes)
    if not isinstance(cfg.data.label_map, dict) or set(cfg.data.label_map.values()) != {0, 1}:
        raise ValueError("invalid data.label_map; mapped classes must be exactly 0 and 1")
    if any(not isinstance(key, int) or not isinstance(value, int) for key, value in cfg.data.label_map.items()):
        raise ValueError("invalid data.label_map; keys and values must be integers")
    if not isinstance(cfg.data.ignore_values, list) or any(
        not isinstance(value, int) for value in cfg.data.ignore_values
    ):
        raise ValueError("data.ignore_values must be a list of integers")
    if len(set(cfg.data.ignore_values)) != len(cfg.data.ignore_values):
        raise ValueError("data.ignore_values must be unique")
    if set(cfg.data.label_map) & set(cfg.data.ignore_values):
        raise ValueError("data.label_map and data.ignore_values must not overlap")
    if not isinstance(cfg.loss.ignore_index, int) or isinstance(cfg.loss.ignore_index, bool):
        raise ValueError("loss.ignore_index must be an integer")
    if cfg.loss.ignore_index in {0, 1}:
        raise ValueError("loss.ignore_index must differ from trainable classes")
    if not isinstance(cfg.data.split_ratios, list) or len(cfg.data.split_ratios) != 3:
        raise ValueError("data.split_ratios must contain three non-negative values")
    if any(not _is_number(value) or value < 0 for value in cfg.data.split_ratios):
        raise ValueError("data.split_ratios must contain three non-negative values")
    if abs(sum(cfg.data.split_ratios) - 1.0) > 1e-9:
        raise ValueError("data.split_ratios must sum to 1")

    for key, values in (
        ("data.normalize_mean", cfg.data.normalize_mean),
        ("data.normalize_std", cfg.data.normalize_std),
    ):
        if not isinstance(values, list) or len(values) != 3 or any(not _is_number(value) for value in values):
            raise ValueError(f"{key} must contain three numbers")
    if any(value <= 0 for value in cfg.data.normalize_std):
        raise ValueError("data.normalize_std values must be positive")
    if not _is_number(cfg.data.horizontal_flip) or not 0 <= cfg.data.horizontal_flip <= 1:
        raise ValueError("data.horizontal_flip must be between 0 and 1")
    if not isinstance(cfg.data.num_workers, int) or isinstance(cfg.data.num_workers, bool) or cfg.data.num_workers < 0:
        raise ValueError("data.num_workers must be a non-negative integer")
    if not isinstance(cfg.data.pin_memory, bool):
        raise ValueError("data.pin_memory must be a boolean")
    _validate_seed("data.seed", cfg.data.seed)
    for key, value in (
        ("data.max_train_samples", cfg.data.max_train_samples),
        ("data.max_valid_samples", cfg.data.max_valid_samples),
        ("data.max_test_samples", cfg.data.max_test_samples),
    ):
        _positive_optional_int(key, value)

    _nonempty_string("model.name", cfg.model.name)
    if cfg.model.factory is None and cfg.model.name not in SUPPORTED_MODEL_NAMES:
        raise ValueError(f"model.name is not registered: {cfg.model.name}")
    if cfg.model.factory is not None:
        _nonempty_string("model.factory", cfg.model.factory)
    _positive_int("model.in_channels", cfg.model.in_channels)
    if not isinstance(cfg.model.params, dict) or any(not isinstance(key, str) for key in cfg.model.params):
        raise ValueError("model.params must be a mapping with string keys")
    if "base_channels" in cfg.model.params:
        _positive_int("model.params.base_channels", cfg.model.params["base_channels"])
    if not isinstance(cfg.model.pretrained, bool):
        raise ValueError("model.pretrained must be a boolean")
    if cfg.loss.name not in ("cross_entropy", "cross_entropy_dice"):
        raise ValueError("loss.name must be cross_entropy or cross_entropy_dice")
    if not _is_number(cfg.loss.cross_entropy_weight) or cfg.loss.cross_entropy_weight < 0:
        raise ValueError("loss.cross_entropy_weight must be non-negative")
    if not _is_number(cfg.loss.dice_weight) or cfg.loss.dice_weight < 0:
        raise ValueError("loss.dice_weight must be non-negative")
    if cfg.loss.cross_entropy_weight + cfg.loss.dice_weight <= 0:
        raise ValueError("at least one loss weight must be positive")

    _positive_int("train.epochs", cfg.train.epochs)
    _positive_int("train.batch_size", cfg.train.batch_size)
    if not _is_number(cfg.train.lr) or cfg.train.lr <= 0:
        raise ValueError("train.lr must be positive")
    if not _is_number(cfg.train.weight_decay) or cfg.train.weight_decay < 0:
        raise ValueError("train.weight_decay must be non-negative")
    if not _is_number(cfg.train.grad_clip) or cfg.train.grad_clip < 0:
        raise ValueError("train.grad_clip must be non-negative")
    if cfg.train.optimizer != "adamw":
        raise ValueError("train.optimizer must be adamw")
    if cfg.train.scheduler not in ("none", "cosine"):
        raise ValueError("train.scheduler must be none or cosine")
    _validate_seed("train.seed", cfg.train.seed)
    if not isinstance(cfg.train.amp, bool):
        raise ValueError("train.amp must be a boolean")
    _nonempty_string("train.best_metric", cfg.train.best_metric)
    valid_best_metric = cfg.train.best_metric in {"mean_iou", "mean_dice", "pixel_accuracy"} or (
        cfg.train.best_metric.startswith("class_iou:") and bool(cfg.train.best_metric.removeprefix("class_iou:"))
    )
    if not valid_best_metric:
        raise ValueError(
            "train.best_metric must be mean_iou, mean_dice, pixel_accuracy, or class_iou:<name>"
        )
    if cfg.device not in ("auto", "cpu", "cuda", "mps"):
        raise ValueError("device must be auto, cpu, cuda, or mps")
    if cfg.log_level not in ("debug", "info", "warning", "error", "critical"):
        raise ValueError("log_level must be a standard logging level")
    if not isinstance(cfg.output_dir, str) or not cfg.output_dir:
        raise ValueError("output_dir must be a non-empty string")
    if cfg.run_name is not None and (not isinstance(cfg.run_name, str) or not cfg.run_name):
        raise ValueError("run_name must be null or a non-empty string")


def to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    """Convert a configuration to serializable built-in containers."""
    return dataclasses.asdict(cfg)


def dump_config(cfg: ExperimentConfig, path: str | Path) -> None:
    """Write a resolved configuration as readable YAML."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(to_dict(cfg), sort_keys=False))

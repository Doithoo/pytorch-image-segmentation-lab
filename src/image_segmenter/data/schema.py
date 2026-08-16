"""Canonical class names, identifiers, and display colors."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ClassDefinition:
    """One trainable class in a segmentation label space."""

    id: int
    name: str
    color: tuple[int, int, int]

    def __post_init__(self) -> None:
        if not isinstance(self.id, int) or isinstance(self.id, bool) or self.id < 0:
            raise ValueError("class id must be a non-negative integer")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("class name must be a non-empty string")
        if (
            not isinstance(self.color, tuple)
            or len(self.color) != 3
            or any(not isinstance(channel, int) or isinstance(channel, bool) for channel in self.color)
            or any(channel < 0 or channel > 255 for channel in self.color)
        ):
            raise ValueError("class color must be an RGB tuple with integer values from 0 to 255")


@dataclass(frozen=True)
class LabelSchema:
    """Canonical semantics shared by data, metrics, and visualization."""

    classes: tuple[ClassDefinition, ...]
    ignore_index: int = 255

    def __post_init__(self) -> None:
        if not isinstance(self.classes, tuple) or not self.classes:
            raise ValueError("classes must be a non-empty tuple")
        if any(not isinstance(item, ClassDefinition) for item in self.classes):
            raise ValueError("classes must contain ClassDefinition values")
        expected_ids = list(range(len(self.classes)))
        actual_ids = [item.id for item in self.classes]
        if actual_ids != expected_ids:
            raise ValueError(f"class ids must be contiguous from 0; got {actual_ids}")
        names = [item.name for item in self.classes]
        if len(set(names)) != len(names):
            raise ValueError("class names must be unique")
        colors = [item.color for item in self.classes]
        if len(set(colors)) != len(colors):
            raise ValueError("class colors must be unique")
        if not isinstance(self.ignore_index, int) or isinstance(self.ignore_index, bool):
            raise ValueError("ignore_index must be an integer")
        if self.ignore_index in actual_ids:
            raise ValueError("ignore_index must differ from every class id")

    @property
    def num_classes(self) -> int:
        """Return the number of trainable classes."""
        return len(self.classes)

    def class_id(self, name: str) -> int:
        """Resolve a class name to its canonical integer id."""
        for item in self.classes:
            if item.name == name:
                return item.id
        raise KeyError(f"unknown class name: {name}")

    def to_dict(self) -> dict[str, Any]:
        """Convert the schema to YAML-safe built-in containers."""
        return {
            "classes": [{"id": item.id, "name": item.name, "color": list(item.color)} for item in self.classes],
            "ignore_index": self.ignore_index,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> LabelSchema:
        """Build and validate a schema from parsed YAML data."""
        if not isinstance(raw, dict):
            raise ValueError("label schema must be a mapping")
        unknown = sorted(set(raw) - {"classes", "ignore_index"})
        if unknown:
            raise ValueError(f"unknown label schema key: {unknown[0]}")
        raw_classes = raw.get("classes")
        if not isinstance(raw_classes, list):
            raise ValueError("label schema classes must be a list")

        definitions: list[ClassDefinition] = []
        for index, item in enumerate(raw_classes):
            if not isinstance(item, dict):
                raise ValueError(f"label schema class {index} must be a mapping")
            unknown_class_keys = sorted(set(item) - {"id", "name", "color"})
            if unknown_class_keys:
                raise ValueError(f"unknown class key: {unknown_class_keys[0]}")
            if set(item) != {"id", "name", "color"}:
                raise ValueError(f"label schema class {index} requires id, name, and color")
            color = item["color"]
            if not isinstance(color, list | tuple):
                raise ValueError(f"label schema class {index} color must be a sequence")
            definitions.append(ClassDefinition(id=item["id"], name=item["name"], color=tuple(color)))

        return cls(classes=tuple(definitions), ignore_index=raw.get("ignore_index", 255))

    @classmethod
    def read_yaml(cls, path: str | Path) -> LabelSchema:
        """Read a label schema from a YAML file."""
        schema_path = Path(path)
        loaded = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
        return cls.from_dict(loaded)

    def write_yaml(self, path: str | Path) -> None:
        """Atomically write this schema as readable YAML."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=output.parent,
                prefix=f".{output.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                yaml.safe_dump(self.to_dict(), handle, sort_keys=False)
                temporary_path = Path(handle.name)
            os.replace(temporary_path, output)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

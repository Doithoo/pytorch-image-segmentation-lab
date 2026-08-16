"""Versioned metadata for a prepared segmentation dataset."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .schema import LabelSchema

DATASET_METADATA_SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _schema_hash(schema: LabelSchema) -> str:
    encoded = json.dumps(schema.to_dict(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PreparedDatasetMetadata:
    """Portable split locations, provenance, and label semantics."""

    provider: str
    split_method: str
    seed: int | None
    data_dir: str
    manifests: dict[str, str]
    split_counts: dict[str, int]
    manifest_sha256: dict[str, str]
    label_schema: LabelSchema
    label_schema_sha256: str
    label_map: dict[int, int] = field(default_factory=dict)
    ignore_values: tuple[int, ...] = ()
    audit: dict[str, Any] = field(default_factory=dict)
    schema_version: int = DATASET_METADATA_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DATASET_METADATA_SCHEMA_VERSION:
            raise ValueError(
                f"dataset metadata schema {self.schema_version} is not supported; "
                f"expected {DATASET_METADATA_SCHEMA_VERSION}"
            )
        if not self.provider or not self.split_method:
            raise ValueError("dataset provider and split_method must be non-empty")
        if Path(self.data_dir).is_absolute():
            raise ValueError("dataset data_dir must be portable and relative")
        if not self.manifests or any(Path(value).is_absolute() for value in self.manifests.values()):
            raise ValueError("dataset manifest paths must be non-empty and relative")
        keys = set(self.manifests)
        if set(self.split_counts) != keys or set(self.manifest_sha256) != keys:
            raise ValueError("manifest paths, counts, and hashes must describe the same splits")
        if any(not isinstance(count, int) or isinstance(count, bool) or count < 0 for count in self.split_counts.values()):
            raise ValueError("dataset split counts must be non-negative integers")
        if self.seed is not None and (not isinstance(self.seed, int) or isinstance(self.seed, bool)):
            raise ValueError("dataset seed must be an integer or null")

    @classmethod
    def create(
        cls,
        *,
        provider: str,
        split_method: str,
        seed: int | None,
        data_dir: str,
        manifests: dict[str, str],
        split_counts: dict[str, int],
        label_schema: LabelSchema,
        root: str | Path,
        label_map: dict[int, int] | None = None,
        ignore_values: tuple[int, ...] = (),
        audit: dict[str, Any] | None = None,
    ) -> PreparedDatasetMetadata:
        """Create metadata and hash the referenced manifests."""
        metadata_root = Path(root)
        hashes = {split: _sha256(metadata_root / relative) for split, relative in manifests.items()}
        return cls(
            provider=provider,
            split_method=split_method,
            seed=seed,
            data_dir=data_dir,
            manifests=dict(manifests),
            split_counts=dict(split_counts),
            manifest_sha256=hashes,
            label_schema=label_schema,
            label_schema_sha256=_schema_hash(label_schema),
            label_map=dict(label_map or {}),
            ignore_values=tuple(ignore_values),
            audit=dict(audit or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to YAML-safe built-in containers."""
        return {
            "schema_version": self.schema_version,
            "provider": self.provider,
            "split_method": self.split_method,
            "seed": self.seed,
            "data_dir": self.data_dir,
            "manifests": dict(self.manifests),
            "split_counts": dict(self.split_counts),
            "manifest_sha256": dict(self.manifest_sha256),
            "label_schema": self.label_schema.to_dict(),
            "label_schema_sha256": self.label_schema_sha256,
            "label_map": dict(self.label_map),
            "ignore_values": list(self.ignore_values),
            "audit": dict(self.audit),
        }


def write_dataset_metadata(path: str | Path, metadata: PreparedDatasetMetadata) -> Path:
    """Atomically write prepared-dataset metadata."""
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
            yaml.safe_dump(metadata.to_dict(), handle, sort_keys=False)
            temporary_path = Path(handle.name)
        os.replace(temporary_path, output)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return output


def read_dataset_metadata(path: str | Path) -> PreparedDatasetMetadata:
    """Read metadata and reject stale or tampered schema/manifests."""
    metadata_path = Path(path)
    loaded = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("dataset metadata root must be a mapping")
    version = loaded.get("schema_version")
    if version != DATASET_METADATA_SCHEMA_VERSION:
        raise ValueError(
            f"dataset metadata schema {version} is not supported; expected {DATASET_METADATA_SCHEMA_VERSION}"
        )
    required = {
        "provider",
        "split_method",
        "seed",
        "data_dir",
        "manifests",
        "split_counts",
        "manifest_sha256",
        "label_schema",
        "label_schema_sha256",
    }
    missing = sorted(required - set(loaded))
    if missing:
        raise ValueError(f"dataset metadata is missing required keys: {missing}")
    schema = LabelSchema.from_dict(loaded["label_schema"])
    expected_schema_hash = _schema_hash(schema)
    if loaded["label_schema_sha256"] != expected_schema_hash:
        raise ValueError("label schema hash mismatch in dataset metadata")
    manifests = {str(key): str(value) for key, value in loaded["manifests"].items()}
    stored_hashes = {str(key): str(value) for key, value in loaded["manifest_sha256"].items()}
    for split, relative in manifests.items():
        manifest_path = metadata_path.parent / relative
        if not manifest_path.is_file():
            raise FileNotFoundError(f"prepared manifest not found for {split}: {manifest_path}")
        actual_hash = _sha256(manifest_path)
        if stored_hashes.get(split) != actual_hash:
            raise ValueError(f"manifest hash mismatch for split {split}")
    return PreparedDatasetMetadata(
        provider=loaded["provider"],
        split_method=loaded["split_method"],
        seed=loaded["seed"],
        data_dir=loaded["data_dir"],
        manifests=manifests,
        split_counts={str(key): int(value) for key, value in loaded["split_counts"].items()},
        manifest_sha256=stored_hashes,
        label_schema=schema,
        label_schema_sha256=loaded["label_schema_sha256"],
        label_map={int(key): int(value) for key, value in loaded.get("label_map", {}).items()},
        ignore_values=tuple(int(value) for value in loaded.get("ignore_values", [])),
        audit=dict(loaded.get("audit", {})),
        schema_version=version,
    )

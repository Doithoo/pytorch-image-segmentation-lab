"""Data preparation and loading public API."""

from .dataset import SegmentationDataset, map_mask
from .manifest import ManifestError, ManifestRow, prepare_oxford_manifests, prepare_paired_manifests
from .masks import MaskDecoder
from .metadata import PreparedDatasetMetadata, read_dataset_metadata, write_dataset_metadata
from .registry import (
    DatasetSpec,
    DatasetStatus,
    available_datasets,
    dataset_status,
    get_dataset_spec,
    prepare_dataset,
    register_dataset,
)
from .schema import ClassDefinition, LabelSchema
from .transforms import ImageTransform, PairedTransform, build_transform

__all__ = [
    "ClassDefinition",
    "DatasetSpec",
    "DatasetStatus",
    "LabelSchema",
    "ManifestError",
    "ManifestRow",
    "MaskDecoder",
    "PreparedDatasetMetadata",
    "ImageTransform",
    "PairedTransform",
    "SegmentationDataset",
    "available_datasets",
    "build_transform",
    "dataset_status",
    "get_dataset_spec",
    "map_mask",
    "prepare_oxford_manifests",
    "prepare_paired_manifests",
    "prepare_dataset",
    "read_dataset_metadata",
    "register_dataset",
    "write_dataset_metadata",
]

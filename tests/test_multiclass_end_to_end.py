import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from image_segmenter.config import config_from_dict
from image_segmenter.data import (
    ClassDefinition,
    LabelSchema,
    PreparedDatasetMetadata,
    write_dataset_metadata,
)
from image_segmenter.evaluation.evaluate import evaluate_checkpoint
from image_segmenter.inference import Predictor
from image_segmenter.training.checkpoint import load_checkpoint
from image_segmenter.training.train import train_from_config


def _prepared_three_class_dataset(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    prepared = tmp_path / "prepared"
    source.mkdir()
    prepared.mkdir()
    split_ids = {"train": ("train-a", "train-b"), "valid": ("valid-a",), "test": ("test-a",)}
    manifests: dict[str, str] = {}
    counts: dict[str, int] = {}
    for split, sample_ids in split_ids.items():
        manifest = prepared / f"{split}.csv"
        manifests[split] = manifest.name
        counts[split] = len(sample_ids)
        with manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("sample_id", "image_path", "mask_path", "split", "width", "height"),
                lineterminator="\n",
            )
            writer.writeheader()
            for index, sample_id in enumerate(sample_ids):
                image_path = source / f"{sample_id}.png"
                mask_path = source / f"{sample_id}.mask.png"
                image = np.zeros((16, 16, 3), dtype=np.uint8)
                image[:, :8, 1] = 100 + index
                image[:, 8:, 2] = 180
                mask = np.zeros((16, 16), dtype=np.uint8)
                mask[:, 4:8] = 1
                mask[:, 8:12] = 2
                mask[:2] = 255
                Image.fromarray(image).save(image_path)
                Image.fromarray(mask).save(mask_path)
                writer.writerow(
                    {
                        "sample_id": sample_id,
                        "image_path": Path("..") / "source" / image_path.name,
                        "mask_path": Path("..") / "source" / mask_path.name,
                        "split": split,
                        "width": 16,
                        "height": 16,
                    }
                )
    schema = LabelSchema(
        classes=(
            ClassDefinition(0, "background", (0, 0, 0)),
            ClassDefinition(1, "road", (128, 64, 128)),
            ClassDefinition(2, "vehicle", (0, 0, 142)),
        ),
        ignore_index=255,
    )
    metadata = PreparedDatasetMetadata.create(
        provider="manifest",
        split_method="provided",
        seed=None,
        data_dir="../source",
        manifests=manifests,
        split_counts=counts,
        label_schema=schema,
        root=prepared,
        label_map={0: 0, 1: 1, 2: 2},
        ignore_values=(255,),
    )
    write_dataset_metadata(prepared / "dataset.yaml", metadata)
    return prepared, source / "test-a.png"


def test_three_class_prepare_train_evaluate_predict_flow(tmp_path: Path) -> None:
    prepared, prediction_image = _prepared_three_class_dataset(tmp_path)
    config = config_from_dict(
        {
            "config_version": 2,
            "data": {
                "name": "manifest",
                "manifest_dir": str(prepared),
                "image_size": 16,
                "horizontal_flip": 0.0,
                "num_workers": 0,
                "max_train_samples": None,
                "max_valid_samples": None,
                "max_test_samples": None,
            },
            "model": {
                "name": "unet_small",
                "expected_num_classes": 3,
                "pretrained": False,
                "params": {"base_channels": 2},
            },
            "loss": {"name": "cross_entropy_dice", "cross_entropy_weight": 0.5, "dice_weight": 0.5},
            "train": {"epochs": 1, "batch_size": 2, "best_metric": "class_iou:vehicle"},
            "device": "cpu",
            "output_dir": str(tmp_path / "artifacts"),
            "run_name": "three-class",
        }
    )

    assert train_from_config(config, dry_run=True) is None
    assert not (Path(config.output_dir) / "three-class").exists()
    run_dir = train_from_config(config)
    assert run_dir is not None
    checkpoint = load_checkpoint(run_dir / "best.pt")
    assert checkpoint["label_schema"]["classes"][2]["name"] == "vehicle"
    assert checkpoint["model_metadata"]["architecture"] == "unet_small"

    evaluation = evaluate_checkpoint(run_dir / "best.pt", split="test", device="cpu", sample_limit=1)
    metrics = json.loads((evaluation / "metrics.json").read_text())
    assert "class_iou:vehicle" in metrics["global"]
    assert {row["class_name"] for row in csv.DictReader((evaluation / "per_class_metrics.csv").open())} == {
        "background",
        "road",
        "vehicle",
    }

    predictor = Predictor(run_dir / "best.pt", device="cpu")
    outputs = predictor.predict_path(prediction_image, tmp_path / "prediction.png")
    raw = np.asarray(Image.open(outputs["raw"]))
    assert raw.shape == (16, 16)
    assert set(np.unique(raw)) <= {0, 1, 2}
    assert torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)["label_schema_sha256"]

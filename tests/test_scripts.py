import csv
from pathlib import Path

import numpy as np
from PIL import Image
from scripts.plot_metrics import plot_metrics
from scripts.preview_dataset import create_dataset_preview


def test_create_dataset_preview_renders_four_panels(tmp_path: Path) -> None:
    image_path = tmp_path / "pet.jpg"
    mask_path = tmp_path / "pet.png"
    Image.new("RGB", (16, 12), color=(80, 120, 160)).save(image_path)
    raw_mask = np.full((12, 16), 2, dtype=np.uint8)
    raw_mask[2:10, 4:12] = 1
    raw_mask[1, 3:13] = 3
    Image.fromarray(raw_mask).save(mask_path)
    manifest = tmp_path / "train.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "image_path", "mask_path"])
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "pet",
                "image_path": image_path.name,
                "mask_path": mask_path.name,
            }
        )

    output = create_dataset_preview(manifest, tmp_path / "preview.png", limit=1)

    with Image.open(output) as preview:
        assert preview.mode == "RGB"
        assert preview.size == (64, 36)


def test_plot_metrics_creates_loss_curve(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.csv"
    metrics.write_text(
        "epoch,train_loss,train_pet_iou,valid_loss,valid_pet_iou,lr\n"
        "1,0.9,0.1,1.0,0.08,0.001\n"
        "2,0.6,0.4,0.7,0.35,0.0005\n"
    )

    output = plot_metrics(metrics, tmp_path / "loss.png")

    with Image.open(output) as curve:
        assert curve.width > curve.height
        assert curve.mode in {"RGB", "RGBA"}

"""Checkpoint-only single-image and directory mask prediction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..config import config_from_dict
from ..data.transforms import ImageTransform
from ..evaluation.visualization import save_prediction_outputs
from ..models import create_model
from ..training.checkpoint import checkpoint_label_schema, load_checkpoint
from ..training.train import resolve_device

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png"}


class Predictor:
    """Restore model and preprocessing metadata from one checkpoint."""

    def __init__(self, checkpoint: str | Path, *, device: str = "auto") -> None:
        self.checkpoint_path = Path(checkpoint).resolve()
        saved = load_checkpoint(self.checkpoint_path)
        self.config = config_from_dict(saved["config"])
        self.label_schema = checkpoint_label_schema(saved)
        self.device = resolve_device(device)
        self.model = create_model(
            self.config.model.name,
            factory=self.config.model.factory,
            in_channels=self.config.model.in_channels,
            num_classes=self.label_schema.num_classes,
            pretrained=False,
            params=self.config.model.params,
        )
        self.model.load_state_dict(saved["model_state"])
        self.model.to(self.device).eval()
        self.transform = ImageTransform(
            self.config.data.image_size,
            self.config.data.normalize_mean,
            self.config.data.normalize_std,
            resize_mode=self.config.data.resize_mode,
        )

    def predict_image(self, image: Image.Image) -> np.ndarray:
        """Predict class indices and restore the original image size."""
        original = image.convert("RGB")
        tensor = self.transform(original).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prediction = self.model(tensor).argmax(dim=1).squeeze(0).cpu().numpy()
        if self.config.data.resize_mode == "letterbox":
            target_height, target_width = self.transform.output_size
            scale = min(target_width / original.width, target_height / original.height)
            resized_width = max(1, min(target_width, round(original.width * scale)))
            resized_height = max(1, min(target_height, round(original.height * scale)))
            left = (target_width - resized_width) // 2
            top = (target_height - resized_height) // 2
            prediction = prediction[top : top + resized_height, left : left + resized_width]
        dtype = np.uint8 if self.label_schema.num_classes <= 256 else np.uint16
        restored = Image.fromarray(prediction.astype(dtype)).resize(original.size, resample=Image.Resampling.NEAREST)
        return np.asarray(restored, dtype=dtype)

    def predict_path(self, image_path: str | Path, output: str | Path) -> dict[str, Path]:
        """Predict one image and save raw, color, and overlay outputs."""
        source = Path(image_path).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"input image not found: {source}")
        if source.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"unsupported image file: {source}")
        raw_output = Path(output).with_suffix(".png").resolve()
        if raw_output == source:
            raise ValueError("prediction output would overwrite input image")
        with Image.open(source) as opened:
            image = opened.convert("RGB")
        prediction = self.predict_image(image)
        return save_prediction_outputs(raw_output, image, prediction, schema=self.label_schema)

    def predict_directory(self, input_dir: str | Path, output_dir: str | Path) -> list[dict[str, Path]]:
        """Predict supported files recursively while preserving relative paths."""
        source = Path(input_dir).resolve()
        destination = Path(output_dir).resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"input directory not found: {source}")
        if source == destination:
            raise ValueError("prediction output directory would overwrite input images")
        images = sorted(
            path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        )
        if not images:
            raise ValueError(f"no supported images found in directory: {source}")
        results: list[dict[str, Path]] = []
        for image_path in images:
            relative = image_path.relative_to(source).with_suffix(".png")
            results.append(self.predict_path(image_path, destination / relative))
        return results

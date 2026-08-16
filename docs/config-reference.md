# Configuration Reference

[简体中文](config-reference.zh-CN.md) | [Back to README](../README.md)

All YAML keys map to typed dataclasses in `config.py`. This page explains intent;
use `show-config` for exact current defaults.

## Top-Level Fields

- `device`: `auto`, `cpu`, `cuda`, or `mps`. Auto prefers CUDA, then MPS, then
  CPU. An unavailable explicit device is an error.
- `output_dir`: root for run artifacts.
- `run_name`: stable subdirectory name; generated when omitted.

## Data Fields

- `data.manifest_dir`: directory containing `train.csv`, `valid.csv`, and
  `test.csv`.
- `data.image_size`: square network size, positive and divisible by 16.
- `data.label_map`: raw integer to class index mapping. Oxford uses `{1: 1,
  2: 0}`.
- `data.ignore_values`: raw values mapped to the loss ignore index; Oxford uses
  `[3]`.
- `data.normalize_mean/std`: three image-channel values.
- `data.horizontal_flip`: training-only probability from 0 to 1.
- `data.num_workers`, `pin_memory`: DataLoader controls.
- `data.max_*_samples`: optional bounded subsets for learning runs.

## Model And Loss Fields

- `model.name`: `unet` or `deeplabv3_resnet50`.
- `model.in_channels`: 3 for RGB.
- `model.num_classes`: 2 for background/pet.
- `model.base_channels`: first U-Net width; larger values use more memory.
- `model.pretrained`: advanced DeepLab weight switch; may download weights.
- `loss.cross_entropy_weight`, `loss.dice_weight`: non-negative weights with a
  positive sum.
- `loss.ignore_index`: mapped target value 255.

## Training Fields

- `train.batch_size`, `train.epochs`, `train.lr`, `train.weight_decay`:
  standard optimization controls.
- `train.optimizer`: currently `adamw`.
- `train.scheduler`: `none` or `cosine`.
- `train.seed`: seeds Python, NumPy, PyTorch, workers, and sampling.
- `train.gradient_clip_norm`: optional maximum gradient norm.

## Validate A File

```bash
uv run segment show-config --config configs/reference_unet.yaml
```

See the annotated files in [configs](../configs/README.md). Changes to image
size, normalization, labels, or model construction affect checkpoint
compatibility and should define a new run.

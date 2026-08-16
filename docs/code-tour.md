# Code Tour: Run First, Then Read

[简体中文](code-tour.zh-CN.md) | [Back to README](../README.md)

Trace one image from a manifest row to a prediction instead of reading `src/`
alphabetically.

| Step | Run | Read | Focus |
| --- | --- | --- | --- |
| 1 | example 01 | `data/dataset.py::map_mask` | raw values to target |
| 2 | example 02 | `data/transforms.py` | paired geometry |
| 3 | example 03 | `data/dataset.py` | sample to batch |
| 4 | example 04 | `models/unet.py`, `training/losses.py` | logits and loss |
| 5 | training dry run | `training/train.py`, `trainer.py` | production assembly |
| 6 | example 05 | `inference/predictor.py` | checkpoint to mask |

`manifest.py` validates and records pairs. Dataset opens one row, applies paired
transforms, and maps labels. U-Net only converts image tensors to same-resolution
logits. `train.py` assembles components; Trainer runs epochs and saves best/last.

```bash
uv run segment train --config configs/learning_minimal.yaml --dry-run
```

Evaluation metrics count pixels, evaluate writes reports and selected visuals,
and Predictor handles unlabeled images. On a first read, skip strict config
helpers, Oxford audit details, RNG persistence, argparse declarations, and pixel-
level rendering internals. Return when a concrete question requires them.

Use [configuration flow](configuration-flow.md) for overrides and
[how it works](how-it-works.md) for formulas and design reasons.

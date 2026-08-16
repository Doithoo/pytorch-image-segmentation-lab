# Code Tour

[简体中文](code-tour.zh-CN.md) | [Back to README](../README.md)

The package is deliberately small. Follow one sample from a manifest row to a
prediction rather than reading files alphabetically.

## Data Boundary

`data/manifest.py` validates matching image/mask pairs and writes portable CSV
rows. Oxford logic lives here and in `scripts/download_data.py`; the Dataset
does not know about archive layouts. `data/transforms.py` owns paired geometry.
`data/dataset.py` loads one row and maps raw labels to training indices.

Invariant: masks remain integer class IDs, and raw label 3 becomes ignore index
255 before loss or metrics.

## Model And Loss Boundary

`models/unet.py` is the readable primary model. Four encoder features become
skip connections; decoder blocks upsample and concatenate them. The registry
returns either U-Net or the advanced DeepLab adapter with the same logits
contract.

`training/losses.py` combines cross-entropy and foreground soft Dice while
excluding ignored pixels. It reports components separately so logs expose what
the optimizer is minimizing.

## Training Boundary

`training/train.py` assembles configuration, data loaders, model, optimizer,
scheduler, and device. `training/trainer.py` owns epoch loops and chooses
`best.pt` using validation pet IoU. `training/checkpoint.py` atomically saves
model and resume state with schema validation.

```bash
uv run segment train --config configs/learning_minimal.yaml --dry-run
```

Use this command while reading the three modules: it crosses every boundary
once without hiding behavior behind a notebook state.

## Evaluation And Inference Boundary

`evaluation/metrics.py` accumulates a confusion matrix over valid pixels.
`evaluation/evaluate.py` restores a checkpoint, writes tables, ranks worst
cases, and renders comparisons. `evaluation/visualization.py` keeps raw masks
separate from human-readable colors.

`inference/predictor.py` restores model and preprocessing from the checkpoint,
predicts at network size, then resizes class indices to the original image with
nearest-neighbor interpolation.

## CLI Boundary

`cli.py` parses commands and converts expected errors into concise messages.
Configuration is resolved before expensive work. Read
[Configuration Flow](configuration-flow.md) next, then use the five
[examples](../examples/README.md) as focused entry points.

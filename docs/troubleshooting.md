# Troubleshooting

[简体中文](troubleshooting.zh-CN.md) | [Back to README](../README.md)

Keep the complete error, return to the smallest command, and change one thing at
a time. Changing data, model, and device together hides the cause.

## Installation and commands

| Symptom | Likely cause | First action |
| --- | --- | --- |
| `segment: command not found` | The project is not installed or uv is bypassed | Run `uv sync --extra dev`, then `uv run segment --help` |
| Unsupported Python | Interpreter is outside 3.10–3.12 | Run `uv python pin 3.12` and sync again |
| CUDA/MPS unavailable | Config forces a missing accelerator | Add `--device cpu` before debugging drivers |
| Wrong package imported | Another environment contains an old install | Print `image_segmenter.__file__` through `uv run python` |

## Data and masks

| Symptom | Likely cause | First action |
| --- | --- | --- |
| Missing image or mask | Pair stems differ | Compare `images/a.jpg` with `masks/a.png` |
| Unexpected mask values | Raw values are absent from label/ignore config | Print unique values from one raw mask |
| Empty split | Too few samples or unsuitable ratios | Add samples or adjust `data.split_ratios` |
| Misaligned preview | Source sizes differ or geometry was not paired | Inspect originals and run example 02 |
| New mask values after resize | Bilinear interpolation touched the mask | Use nearest-neighbor interpolation for masks |

## Training

| Symptom | Likely cause | First action |
| --- | --- | --- |
| Dry run fails | Data, labels, output, or loss is disconnected | Run `uv run segment --debug train ...` and inspect the first exception |
| NaN loss | LR is too high or a batch has no valid pixels | Inspect mask values and valid-pixel ratio, then lower LR |
| Loss falls but IoU does not | Background dominates or validation is too small | Inspect overlays, then compare CE with CE+Dice |
| Train improves while validation degrades | Overfitting | Keep the baseline and try flips, a narrower model, or fewer epochs |
| MPS/CUDA OOM | Batch, input, or model is too large | Halve batch size, then image size/base channels |
| Cosine resume rejects new epochs | Saved schedule and new total disagree | Keep the planned total or start a newly planned run |

## Evaluation and prediction

| Symptom | Likely cause | First action |
| --- | --- | --- |
| High pixel accuracy, poor mask | Background dominates | Read pet IoU, Dice, and error overlays |
| Manifest not found | Checkpoint refers to a relative manifest directory | Run from the repository root or restore that manifest set |
| Wrong output size | Network and source image sizes were confused | Use Predictor, which restores source dimensions |
| Color output fails as training data | Color/overlay files are display artifacts | Train only from original single-channel class masks |

When opening an issue, include the complete command, first error, package version,
and resolved config. Add a privacy-safe preview and mask unique values for data
problems; do not upload the complete dataset.

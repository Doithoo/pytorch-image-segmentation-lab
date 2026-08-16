# PyTorch Image Segmentation Lab

[简体中文](README.zh-CN.md)

A small, complete semantic segmentation project for learners who have met
PyTorch tensors and image classification and now want to understand pixel-level
prediction. The main path uses a handwritten U-Net and Oxford-IIIT Pet. Every
step remains visible: paired transforms, raw-label mapping, ignored pixels,
losses, foreground metrics, checkpoints, evaluation, and inference artifacts.

This repository optimizes for learning and reproducibility, not benchmark
leadership. It supports binary semantic segmentation only; instance and
panoptic segmentation are intentionally outside the first release.

## What You Will Build

An input RGB image becomes two logits per pixel: background and pet. Training
uses the official Oxford trimaps with this fixed mapping:

```text
1 -> pet
2 -> background
3 -> ignored/unclassified (training index 255)
```

Raw value 3 is the dataset's unclassified region. It is not a semantic
"boundary class" and contributes to neither loss nor metrics. The best
checkpoint is selected by validation pet IoU. The untouched official test split
is used only for final reporting.

## Quick Start

Requires Python 3.10-3.12 and [uv](https://docs.astral.sh/uv/). The default
learning configuration runs on CPU and does not download pretrained weights.

```bash
git clone https://github.com/Doithoo/pytorch-image-segmentation-lab.git
cd pytorch-image-segmentation-lab
uv sync --extra dev
```

Download and verify Oxford-IIIT Pet, then create deterministic manifests. The
official test membership is preserved; the upstream trainval samples are split
80/20 within each breed into train and validation sets.

```bash
uv run python scripts/download_data.py --data-dir data/raw
uv run segment prepare-data --config configs/learning_minimal.yaml \
  --data-dir data/raw --manifest-dir data/manifests \
  --source-format oxford-pet
uv run python scripts/preview_dataset.py data/manifests/train.csv \
  --output artifacts/dataset_preview.png --limit 4
```

Verify one full optimization batch before starting a run, then train for two
small learning epochs.

```bash
uv run segment train --config configs/learning_minimal.yaml --dry-run
uv run segment train --config configs/learning_minimal.yaml
```

Evaluate a saved checkpoint and predict a mask for one image. Replace the
example paths with those printed by training.

```bash
uv run segment evaluate --checkpoint artifacts/learning-unet/best.pt \
  --split test --plot --device cpu
uv run segment predict --checkpoint artifacts/learning-unet/best.pt \
  --image data/raw/images/Abyssinian_1.jpg \
  --output artifacts/predictions/Abyssinian_1.png --device cpu
```

## Learn In Small Steps

Start with the five runnable files in [examples](examples/README.md), then use
the [learning path](docs/learning-path.md) to move from tensor shapes to a full
experiment. The six-part [tutorial](docs/tutorial/00-basics.md) explains the
same path with concepts, checkpoints, and questions to answer.

Useful maps:

- [How the system works](docs/how-it-works.md)
- [Code tour](docs/code-tour.md)
- [Configuration flow](docs/configuration-flow.md)
- [Complete configuration reference](docs/config-reference.md)
- [Use your own paired data](docs/using-your-data.md)
- [Run controlled experiments](docs/experiments.md)
- [Measured reference run](docs/reference-run.md)

## Outputs You Can Inspect

Training writes the resolved configuration, environment metadata, epoch
metrics, and atomic `best.pt`/`last.pt` checkpoints under one run directory.
Evaluation adds global and per-image tables, sample comparisons, worst cases,
and an optional loss curve. Prediction keeps machine-readable class-index masks
separate from color masks and overlays.

The primary score is pet foreground IoU. Pixel accuracy can look excellent on
background-heavy images while pet masks remain poor, so it never selects a
checkpoint. A per-image sample with no foreground in both target and prediction
has undefined IoU/Dice; it is counted and excluded from per-image means rather
than silently treated as perfect.

## Configurations

- `configs/learning_minimal.yaml`: bounded CPU-first subset, 128 px images,
  two epochs, cross-entropy only.
- `configs/reference_unet.yaml`: measured 25-epoch handwritten U-Net run.
- `configs/deeplabv3_resnet50.yaml`: advanced torchvision comparison; pretrained
  weights may require a separate network download.

Configuration precedence is `defaults < YAML < --set < dedicated CLI option`.
Unknown keys and invalid combinations fail before expensive work. Inspect the
effective values with:

```bash
uv run segment show-config --config configs/learning_minimal.yaml
uv run segment --help
```

## Reproducibility And Limits

Manifests record split provenance and SHA-256 checksums. Checkpoints store the
effective configuration, label mapping, preprocessing, manifest checksums,
optimizer state, and random state. Seeds reduce variation, but device kernels
and PyTorch versions can still change floating-point results; compare trends
and reported conditions, not only one decimal score.

The reference result is evidence for this implementation and recorded
environment, not a promise for every machine. Oxford-IIIT Pet images and
annotations retain their upstream license and attribution; they are downloaded
locally and are not committed to this repository.

## Development

```bash
uv run ruff check .
uv run pytest
uv build
uv run twine check dist/*
```

Read [CONTRIBUTING.md](CONTRIBUTING.md) before proposing a change. Tests use
synthetic local fixtures and do not download datasets or pretrained models.

Released under the [MIT License](LICENSE).

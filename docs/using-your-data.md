# Using Your Data

[简体中文](using-your-data.zh-CN.md) | [Back to README](../README.md)

Custom data uses the same manifest-backed Dataset as Oxford. The preparation
command only assumes paired filenames and explicit mask semantics.

## Arrange Pairs

```text
my-data/
|-- images/
|   |-- sample-001.jpg
|   `-- sample-002.png
`-- masks/
    |-- sample-001.png
    `-- sample-002.png
```

Image and mask stems must match exactly. Nested paths are allowed only when the
relative stems still match. Masks must be single-channel integer images, not
colorized screenshots or alpha overlays.

## Define Label Semantics

Inspect unique pixel values before editing configuration. `data.label_map`
maps raw values to contiguous class indices; `data.ignore_values` lists pixels
that contribute to neither loss nor metrics. Every raw value must be accounted
for, otherwise loading fails instead of silently treating it as background.

This first release is binary: model class 0 is background and class 1 is
foreground. Multiclass, instance IDs, polygons, and soft masks require code
changes and are not accepted by the default pipeline.

## Prepare Manifests

```bash
uv run segment prepare-data --config configs/learning_minimal.yaml \
  --data-dir my-data --manifest-dir my-manifests \
  --source-format paired
uv run python scripts/preview_dataset.py my-manifests/train.csv \
  --output artifacts/my-data-preview.png --limit 4
```

Generic paired data uses deterministic seeded train/validation/test ratios from
the config. Preparation validates missing pairs, duplicate content, unreadable
files, image/mask size mismatches, and unexpected mask values. Read
`source.yaml` and `summary.txt` before training.

## Adapt The Preview

The bundled preview uses Oxford's `1/2/3` raw palette. For a different raw
encoding, first use the generated manifests and Dataset tests, then adapt the
small mapping constants in `scripts/preview_dataset.py`. Never use the preview's
rendered output as a training mask.

## Train Safely

Copy `learning_minimal.yaml`, set `data.manifest_dir`, label mapping, and ignore
values, then run `show-config` and `--dry-run`. Start with a tiny subset. Verify
that image and target shapes align and mapped values are only `0`, `1`, and the
ignore index before increasing epochs.

Keep a final test split isolated from tuning. If your source has groups such as
patients, videos, or repeated subjects, write a group-aware splitter; random
image splits can leak nearly identical content and inflate results.

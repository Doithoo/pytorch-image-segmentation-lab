# Package Guide

[简体中文](README.zh-CN.md) | [Code tour](../../docs/code-tour.md)

`image_segmenter` is the installed production package. Dependencies point
inward from CLI orchestration to focused data, model, training, evaluation, and
inference modules.

## Start Here

Read `config.py`, then `data/dataset.py`, `models/unet.py`, and
`training/losses.py`. These define the contracts passed through the rest of the
system. Run the dry-run while tracing calls.

```bash
uv run segment train --config configs/learning_minimal.yaml --dry-run
```

## Module Ownership

- `data`: validation, manifests, transforms, loading, and raw-label mapping.
- `models`: handwritten U-Net, DeepLab adapter, and model registry.
- `training`: loss, loop, metrics writing, checkpoints, resume, and device.
- `evaluation`: metrics tables, plots, and diagnostic visualizations.
- `inference`: checkpoint-only prediction and output preservation.
- `cli.py`: argument parsing and user-facing error boundary.

## Change Rules

Keep masks as class indices, keep model outputs as logits, and centralize paired
geometry. A checkpoint contract change requires a schema decision and migration
or explicit rejection. Shared trainer behavior must not branch on a particular
model name.

## Public Surface

The stable beginner surface is the `segment` CLI and documented module classes.
Underscore-prefixed helpers are internal. Add tests at the owning boundary and
an end-to-end test when a change crosses multiple modules.

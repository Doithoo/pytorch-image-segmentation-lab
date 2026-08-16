# Data and Result Utilities

[简体中文](README.zh-CN.md) | [Back to README](../README.md)

Run all scripts from the repository root through `uv run`.

```bash
uv run python scripts/download_data.py --data-dir data/raw
uv run python scripts/download_data.py --data-dir data/raw --verify-only
```

The downloader uses partial files, fixed SHA-256 values, and safe extraction.
Use `--force` only when replacing the cache deliberately.

```bash
uv run python scripts/preview_dataset.py data/manifests/train.csv \
  --output artifacts/dataset_preview.png --limit 4
```

Open the preview and inspect alignment; successful execution alone does not
prove labels are correct.

```bash
uv run python scripts/plot_metrics.py artifacts/first-unet/metrics.csv \
  --output artifacts/first-unet/loss_curve.png
```

Interpret loss curves with validation pet IoU and overlays, not alone.

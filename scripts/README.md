# Utility Scripts

These scripts keep dataset setup and artifact rendering visible to learners.
Run them from the repository root with the project environment active.

## Download Oxford-IIIT Pet

```bash
uv run python scripts/download_data.py --data-dir data/raw
```

Downloads are written atomically to `data/downloads`. The installer verifies
both SHA-256 checksums before creating `data/raw/images`, `data/raw/masks`, and
`data/raw/annotations`. Use `--verify-only` to check cached archives without
network access or installation. Use `--force` only when you intend to download
fresh copies.

## Preview A Manifest

```bash
uv run python scripts/preview_dataset.py data/manifests/train.csv \
  --output artifacts/dataset_preview.png --limit 4
```

Each row shows the source image, raw Oxford trimap, mapped training mask, and
foreground overlay. The script reads source files but never changes them.

## Plot Training Metrics

```bash
uv run python scripts/plot_metrics.py artifacts/my-run/metrics.csv \
  --output artifacts/my-run/loss_curve.png
```

The plot contains the train and validation loss recorded after each epoch.

See [the Simplified Chinese guide](README.zh-CN.md) for the Chinese version.

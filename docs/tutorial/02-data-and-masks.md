# Tutorial 02: Data And Masks

[简体中文](02-data-and-masks.zh-CN.md) | [Tutorial index](../learning-path.md)

Segmentation quality begins with aligned, correctly interpreted labels.

## Download And Verify

```bash
uv run python scripts/download_data.py --data-dir data/raw
uv run python scripts/download_data.py --data-dir data/raw --verify-only
```

Downloads first go to `.part` files. Only a complete matching SHA-256 archive
replaces the cache. Extraction rejects paths that escape the destination.

## Preserve The Official Test Set

```bash
uv run segment prepare-data --config configs/learning_minimal.yaml \
  --data-dir data/raw --manifest-dir data/manifests \
  --source-format oxford-pet
```

The official test list is copied unchanged. Upstream trainval samples are split
80/20 inside every breed. Read `source.yaml`, `summary.txt`, and CSV rows; their
hashes bind later checkpoints to exact membership.

## Inspect Raw And Mapped Masks

```bash
uv run python scripts/preview_dataset.py data/manifests/train.csv \
  --output artifacts/dataset_preview.png --limit 4
uv run python examples/02_paired_transform.py
```

Check that raw values 1, 2, and 3 become pet 1, background 0, and ignored 255.
Confirm flips and resizing keep image boundaries aligned with mask boundaries.

## Detect Leakage And Corruption

Exact duplicate content across splits is an error. Missing pairs, unreadable
files, size mismatches, and unknown mask values also fail early. These checks
cannot detect semantically related subjects; custom medical or video data needs
group-aware splitting.

Next: [Tutorial 03](03-unet.md).

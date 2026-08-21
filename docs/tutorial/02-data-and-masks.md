# Tutorial 02: Trust the Masks Before Training

[简体中文](02-data-and-masks.zh-CN.md) | [Tutorial index](README.md)

A segmentation model will faithfully learn incorrect labels. Spend ten minutes
on images, masks, and membership before a long run.

## Download, verify, and record membership

```bash
uv run python scripts/download_data.py --data-dir data/raw
uv run python scripts/download_data.py --data-dir data/raw --verify-only
```

Downloads use partial files and pinned SHA-256 values. Safe extraction rejects
paths escaping the destination and rejects links.

Create manifests:

```bash
uv run segment prepare-data --config configs/learning_minimal.yaml \
  --data-dir data/raw --manifest-dir data/manifests \
  --source-format oxford-pet
```

The official test list remains untouched; trainval is divided 80/20 within each
breed. CSV manifests make membership explicit and portable. `source.yaml` and
`summary.txt` record semantics, counts, source, and hashes.

## Map trimaps to training targets

Oxford `1/2/3` means pet/background/unclassified border and maps to `1/0/255`.
Index 255 is filtered by cross-entropy, Dice, and metrics. It is not a third
model class.

```bash
uv run python scripts/preview_dataset.py data/manifests/train.csv \
  --output artifacts/dataset_preview.png --limit 4
```

![Images, raw trimaps, mapped masks, and overlays](../assets/reference-unet-dataset-preview.png)

Inspect every panel: valid image, sensible raw boundary, correct mapped colors,
and aligned overlay.

## Keep geometry paired

```bash
uv run python examples/02_paired_transform.py
```

Sample a random flip once and apply it to both image and mask. Resize RGB with
bilinear interpolation and class masks with nearest-neighbor interpolation.
The default `stretch` mode maps pairs directly to a fixed shape. For
aspect-ratio-sensitive data, use `resize_mode: letterbox`; mask padding receives
the ignore index. Optional `crop_size` applies one shared random training crop.

Preparation catches missing pairs, unreadable files, size mismatches,
multi-channel masks, unknown values, exact duplicate pairs, and empty splits.
It cannot detect related patients or video frames across splits; group those
domains before splitting.

Print unique values from a raw mask and inspect `map_mask` in
`data/dataset.py`. Explain why an unknown value raises instead of silently
becoming background. Continue to [U-Net](03-unet.md) or the
[custom-data guide](../guides/using-your-data.md).

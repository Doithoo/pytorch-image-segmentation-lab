# Use Your Own Image/Mask Pairs

[简体中文](using-your-data.zh-CN.md) | [Documentation](../README.md)

The generic path supports same-stem image/mask pairs and contiguous multiclass labels.
Validate a small bounded copy before a full run.

```text
my-data/
|-- images/sample-001.jpg
`-- masks/sample-001.png
```

Print unique values from several raw masks. Put every possible raw value in
`data.label_map` or `data.ignore_values`; unknown values fail instead of silently
becoming background. Keep `loss.ignore_index` different from every mapped class.
For a three-class paired dataset, use a contiguous mapping such as
`{0: 0, 128: 1, 255: 2}`, set `model.expected_num_classes: 3`, choose an unused
`loss.ignore_index`, and select a general metric such as `train.best_metric:
mean_iou`. Generated metadata uses `class_0`, `class_1`, and so on; use explicit
metadata or a Dataset factory when domain names and fixed colors are required.


Create a separate config and manifest directory, then prepare paired data:

```bash
uv run segment prepare-data --config configs/my_data.yaml \
  --data-dir my-data --manifest-dir data/my-manifests \
  --source-format paired
```

Preparation rejects missing pairs, size/value problems, duplicates, and ratios
that produce empty train/valid/test splits. Inspect `summary.txt` and CSV rows.

Run one CPU batch before training:

```bash
uv run segment train --config configs/my_data.yaml --dry-run --device cpu
```

Check image, target, and `[B,K,H,W]` logits shapes; mapped values; and valid-pixel
ratio. The default `stretch` mode maps pairs directly to the configured shape.
Use `resize_mode: letterbox` to preserve aspect ratio with ignored mask padding;
optional `crop_size` applies a shared random training crop.

Random image splitting is unsafe for related patients, video frames, bursts, or
shared sources. Group those entities before assigning splits.

Run a short named baseline, inspect validation overlays, and diagnose labels,
membership, shape distortion, and foreground size before increasing model
complexity. See [troubleshooting](troubleshooting.md) and
[experiments](experiments.md).

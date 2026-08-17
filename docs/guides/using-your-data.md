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
becoming background. Keep `loss.ignore_index` different from classes 0 and 1.

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

Check image, target, logits shapes; mapped values; and valid-pixel ratio. The
built-in transform stretches pairs to a square. Aspect-ratio-sensitive domains
may require a tested paired padding/cropping transform.

Random image splitting is unsafe for related patients, video frames, bursts, or
shared sources. Group those entities before assigning splits.

Run a short named baseline, inspect validation overlays, and diagnose labels,
membership, shape distortion, and foreground size before increasing model
complexity. See [troubleshooting](troubleshooting.md) and
[experiments](experiments.md).

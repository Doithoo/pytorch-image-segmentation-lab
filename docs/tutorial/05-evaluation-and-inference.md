# Tutorial 05: Do Not Let One Score Inspect the Images for You

[简体中文](05-evaluation-and-inference.zh-CN.md) | [Tutorial index](README.md)

Evaluation measures labeled data. Inference produces masks for unlabeled
images. Both restore model and preprocessing from a checkpoint, but their
outputs answer different questions.

```bash
uv run segment evaluate --checkpoint artifacts/first-unet/best.pt \
  --split test --plot --device cpu
```

Evaluation writes summary JSON, per-class and per-image CSV, representative
overlays, worst cases, and an optional loss plot.

For pet foreground, IoU is `TP/(TP+FP+FN)` and Dice is
`2TP/(2TP+FP+FN)`. Pixel accuracy can be dominated by easy background pixels.
Global IoU pools all valid pixels; mean per-image IoU gives each image similar
weight. Report which statistic you use.

When target and prediction both contain no foreground, per-image pet IoU is
undefined. The evaluator counts and excludes that value from the mean instead
of silently assigning a perfect score. An empty target with a false positive
still receives IoU zero.

![Target, prediction, and pixel errors](../assets/reference-unet-overlay.png)

Read an overlay by asking whether errors affect boundaries or the whole object,
whether thin structures disappear, whether background texture causes false
positives, and whether the source annotation itself is suspicious. Worst cases
can reveal data problems as well as model problems.

Use validation to select checkpoints and settings. Evaluate the official test
split only after choosing the final setting; repeatedly selecting by test score
turns test into validation.

```bash
uv run segment predict --checkpoint artifacts/first-unet/best.pt \
  --image data/raw/images/Abyssinian_1.jpg \
  --output artifacts/predictions/Abyssinian_1.png --device cpu
```

Prediction writes a machine-readable 0/1 mask plus color and overlay images.
Predictor restores preprocessing and resizes class indices to source dimensions
with nearest-neighbor interpolation. Never feed display outputs back as target
masks.

A useful report identifies the split and checkpoint, global and per-image
metrics, sample/empty counts, one success and failure image, environment, and a
known limitation. Continue to [experiments](../guides/experiments.md) or
[custom data](../guides/using-your-data.md).

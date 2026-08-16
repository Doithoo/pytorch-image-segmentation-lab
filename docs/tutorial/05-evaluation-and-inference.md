# Tutorial 05: Evaluation And Inference

[简体中文](05-evaluation-and-inference.zh-CN.md) | [Tutorial index](../learning-path.md)

Evaluation measures a labeled split. Inference produces masks for unlabeled
images. Both restore model and preprocessing from the checkpoint.

## Evaluate The Selected Checkpoint

```bash
uv run segment evaluate --checkpoint artifacts/learning-unet/best.pt \
  --split test --plot --device cpu
```

Use validation while developing and the official test split only for the final
selected run. The evaluator writes global metrics, per-class rows, per-image
rows, representative comparisons, worst cases, and an optional loss curve.

## Read Metrics Correctly

Pet IoU is intersection divided by union. Pet Dice is twice intersection divided
by predicted plus target pet pixels. Mean IoU includes background; pixel
accuracy can be dominated by background. Validation pet IoU selects checkpoints.

Dataset-level IoU comes from one accumulated confusion matrix. Mean per-image
IoU gives each eligible image equal weight. Empty target and empty prediction
make per-image pet IoU undefined, so the sample is counted and excluded.

## Diagnose Visually

Comparison grids show image, target, prediction, and errors. Find false
positives, false negatives, true positives, and ignored pixels. Check whether
errors cluster around fur, thin limbs, unusual poses, small pets, or confusing
backgrounds. A curve tells when behavior changed; an overlay tells how.

## Predict New Images

```bash
uv run segment predict --checkpoint artifacts/learning-unet/best.pt \
  --image data/raw/images/Abyssinian_1.jpg \
  --output artifacts/predictions/Abyssinian_1.png --device cpu
```

The raw output PNG contains class indices 0 and 1. `.color.png` is for human
inspection; `.overlay.png` blends pet pixels onto the original. Never feed a
color or overlay output back as a target mask.

## Continue Learning

Write one hypothesis using [Experiments](../experiments.md), or adapt a binary
paired dataset with [Using Your Data](../using-your-data.md). Keep the official
test split isolated and report limitations with every measured result.

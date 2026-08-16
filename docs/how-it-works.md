# How It Works

[简体中文](how-it-works.zh-CN.md) | [Back to README](../README.md)

Semantic segmentation assigns one class to every pixel. This project keeps the
complete path explicit so a learner can inspect each representation.

## Data Flow

```text
official archives -> images/masks -> manifests -> paired transform
-> image [B,3,H,W] + target [B,H,W] -> U-Net logits [B,2,H,W]
-> loss/optimizer or argmax -> metrics + mask artifacts
```

Manifests separate dataset preparation from training. Paths may be relative,
which makes a prepared dataset movable. Image and mask geometry stays aligned;
photometric normalization applies only to images.

## Label Flow

Oxford raw values are `1=pet`, `2=background`, and `3=unclassified`. The Dataset
maps them to `1`, `0`, and `255`. Cross-entropy receives 255 as `ignore_index`;
Dice and metrics explicitly filter the same pixels.

The model never predicts 255. Its two channels mean background and pet. A raw
trimap, mapped target, logits tensor, predicted class mask, color rendering, and
alpha overlay are distinct objects with distinct purposes.

## Learning Flow

At training time, U-Net produces logits. Cross-entropy rewards the correct
class at each valid pixel. Foreground soft Dice emphasizes overlap and reduces
the influence of large background regions. Gradients update weights only after
the combined loss is computed.

The trainer measures validation pet IoU after every epoch. `last.pt` means most
recent training state; `best.pt` means highest validation pet IoU. These files
can differ.

## Evaluation Flow

The official test split remains untouched until final evaluation. Predictions
and targets update one dataset-level confusion matrix. Per-image metrics are
also written for diagnosis, but their arithmetic mean is not the same statistic
as global IoU.

An image with no target foreground and no predicted foreground has undefined
per-image pet IoU and Dice. The evaluator counts it and excludes it from the
per-image mean. Visual overlays reveal where a numeric score came from.

## Reproduction Flow

The resolved config, source metadata, manifest hashes, run metadata, epoch
metrics, and checkpoint state travel together. Prediction restores input size
and normalization from the checkpoint, preventing a CLI default change from
silently changing old models.

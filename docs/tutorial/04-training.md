# Tutorial 04: Training

[简体中文](04-training.zh-CN.md) | [Tutorial index](../learning-path.md)

Training repeats a small sequence while preserving enough state to explain and
resume it.

## One Batch

```text
load -> forward -> loss -> zero gradients -> backward -> optional clip -> step
```

Cross-entropy uses valid background and pet pixels. Foreground soft Dice uses
pet probability and filters ignored pixels. If a batch has no valid foreground,
the Dice term is inactive rather than rewarding an empty mask.

## Dry Run First

```bash
uv run segment train --config configs/learning_minimal.yaml --dry-run
```

The dry run executes a real optimizer update and reports tensor shapes, mapped
mask values, valid-pixel ratio, loss components, and device. It does not create
a normal run. Fix any problem here before committing hours to epochs.

## Epochs And Validation

```bash
uv run segment train --config configs/learning_minimal.yaml
```

Training batches update weights; validation batches use evaluation mode and no
gradients. After each epoch, metrics append to `metrics.csv`, `last.pt` updates,
and `best.pt` updates only when validation pet IoU improves.

Training loss can fall while validation IoU stalls or declines. This is why the
most recent checkpoint is not automatically the best generalizing checkpoint.

## Resume Exactly

Resume restores model, optimizer, scheduler, epoch, best score, and random
states, including the shuffle sequence. It rejects mismatched model settings,
preprocessing, loss and training semantics, labels, ignore index, or manifest
hashes. Data paths may move and total `train.epochs` may increase; do not point
resume at an unrelated run.

## Read The Artifacts

`config.yaml` is the final configuration. `run.yaml` records environment and
source facts. `metrics.csv` is append-only epoch evidence. A checkpoint is a
trusted project artifact and uses Python serialization; do not load unknown
files from untrusted sources.

Next: [Tutorial 05](05-evaluation-and-inference.md).

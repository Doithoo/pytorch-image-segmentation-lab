# Learning Path

[简体中文](learning-path.zh-CN.md) | [Back to README](../README.md)

Follow the stages in order. Each stage starts with a tiny runnable example and
then moves to the production implementation. Do not begin a long training run
until you can explain the shapes and artifacts from the earlier stages.

## Stage 1: Pixels, Masks, And Logits

Run `examples/01_mask_and_logits.py`. Confirm that an image batch has shape
`[B, 3, H, W]`, logits have `[B, 2, H, W]`, and the target has `[B, H, W]`.
Understand why `argmax(dim=1)` is for prediction, while cross-entropy consumes
logits directly.

```bash
uv run python examples/01_mask_and_logits.py
```

Then read [Tutorial 00](tutorial/00-basics.md) and `data/dataset.py` in the
[code tour](code-tour.md).

## Stage 2: Paired Data

Run examples 02 and 03. A geometric transform must make the same random choice
for image and mask. Images use bilinear interpolation; masks use nearest
neighbor so class IDs cannot become fractional values.

```bash
uv run python examples/02_paired_transform.py
uv run python examples/03_dataloader_batch.py
```

Download Oxford, prepare manifests, and inspect a preview before training. Be
able to explain the raw `1/2/3` values and the mapped `1/0/255` values.

## Stage 3: One Optimization Step

Run example 04 and trace forward, loss, zero-grad, backward, and optimizer step.
Then run the real dry-run command. It exercises the actual Dataset, U-Net,
combined loss, and optimizer without creating a normal experiment directory.

```bash
uv run python examples/04_minimal_training_loop.py
uv run segment train --config configs/learning_minimal.yaml --dry-run
```

Read [Tutorial 03](tutorial/03-unet.md) and
[Tutorial 04](tutorial/04-training.md).

## Stage 4: A Small Experiment

Train the CPU-first configuration. Compare training and validation loss, but
use validation pet IoU to decide which checkpoint generalizes best. Never tune
with the official test split.

```bash
uv run segment train --config configs/learning_minimal.yaml
```

Record a question before changing a variable. Use
[Experiments](experiments.md) to compare controlled runs.

## Stage 5: Evaluation And Inference

Run example 05, evaluate the best checkpoint, and inspect both numeric tables
and overlays. Distinguish global confusion metrics from averages of per-image
metrics. Look at false positives and false negatives, not only one score.

```bash
uv run python examples/05_checkpoint_overlay.py \
  --checkpoint artifacts/learning-unet/best.pt \
  --image data/raw/images/Abyssinian_1.jpg --device cpu
```

Finish with [Tutorial 05](tutorial/05-evaluation-and-inference.md), then adapt
the project through [Using Your Data](using-your-data.md).

## Completion Checklist

You are ready to extend the project when you can explain label 3, paired
transforms, logits versus masks, ignored pixels, pet IoU versus Dice,
validation versus test use, checkpoint contents, and one visual failure case.

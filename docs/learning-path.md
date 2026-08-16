# Image Segmentation Learning Path

[简体中文](learning-path.zh-CN.md) | [Back to README](../README.md)

This 8–12 hour path is for a learner who knows basic tensors, loss, and
gradients but has not completed pixel-wise prediction. Run each small step,
inspect its output, and explain it before starting a longer run.

## 0. Verify the environment

```bash
uv sync --extra dev
uv run segment --version
uv run segment show-config --config configs/learning_minimal.yaml
```

The resolved configuration should show 128 px inputs, bounded samples, two
epochs, a small U-Net, and automatic device selection. Use
`show-config` after overrides instead of reconstructing the final values from
memory.

## 1. Identify the tensors

```bash
uv run python examples/01_mask_and_logits.py
uv run python examples/03_dataloader_batch.py
```

Explain image `[B,3,H,W]`, target `[B,H,W]`, logits `[B,2,H,W]`, and prediction
`[B,H,W]`. Cross-entropy consumes logits and integer targets; `argmax(dim=1)` is
for decisions, not training. Read [tutorial 00](tutorial/00-basics.md) if that
distinction is still unclear.

## 2. Inspect masks before training

```bash
uv run python scripts/download_data.py --data-dir data/raw
uv run segment prepare-data --config configs/learning_minimal.yaml \
  --data-dir data/raw --manifest-dir data/manifests \
  --source-format oxford-pet
uv run python scripts/preview_dataset.py data/manifests/train.csv \
  --output artifacts/dataset_preview.png --limit 4
uv run python examples/02_paired_transform.py
```

Open the preview. Check source image/mask pairing, `1/2/3 -> 1/0/255` mapping,
ignored borders, and geometric alignment. Image resize is bilinear; mask resize
is nearest-neighbor. Read [tutorial 02](tutorial/02-data-and-masks.md).

## 3. Follow one optimization step

```bash
uv run python examples/04_minimal_training_loop.py
uv run segment train --config configs/learning_minimal.yaml --dry-run
```

Trace zero-grad, forward, loss, backward, and optimizer step. Then read
`models/unet.py` from the encoder through skip connections to the two-channel
head. `dry-run OK` proves the production pipeline can update one batch; it does
not create a normal run or measure quality.

Use [tutorial 03](tutorial/03-unet.md) for the model and
[tutorial 04](tutorial/04-training.md) for the loop.

## 4. Complete a small run

```bash
uv run segment train --config configs/learning_minimal.yaml \
  --set run_name first-unet
```

Inspect `config.yaml`, `metrics.csv`, `best.pt`, `last.pt`, and `run.yaml` in
that order. Two epochs are a workflow check, not a competitive result. Explain
why the checkpoint with the highest validation pet IoU may differ from the last
epoch.

## 5. Return from metrics to images

```bash
uv run segment evaluate --checkpoint artifacts/first-unet/best.pt \
  --split test --plot --device cpu
uv run python examples/05_checkpoint_overlay.py \
  --checkpoint artifacts/first-unet/best.pt \
  --image data/raw/images/Abyssinian_1.jpg --device cpu
```

Compare pet IoU, pet Dice, and pixel accuracy, then open sample overlays and
worst cases. Identify false positives, false negatives, and ignored pixels.
Distinguish dataset-global IoU from the mean of per-image IoU. Read
[tutorial 05](tutorial/05-evaluation-and-inference.md).

## 6. Run one controlled comparison

```bash
uv run segment train --config configs/learning_minimal.yaml \
  --set run_name exp-ce
uv run segment train --config configs/learning_minimal.yaml \
  --set loss.cross_entropy_weight 0.5 \
  --set loss.dice_weight 0.5 \
  --set run_name exp-ce-dice
```

Keep manifests, seeds, model width, limits, and epochs fixed. Compare validation
pet IoU, curves, and errors on the same images. A two-epoch, one-seed result
describes these runs; it does not prove a universal Dice-loss rule. Continue
with the [experiment guide](experiments.md) or adapt the
[custom-data guide](using-your-data.md).

You are ready to extend the project when you can explain mask interpolation,
label 3 versus index 255, logits versus targets, best versus last checkpoint,
validation versus test use, global versus per-image IoU, and one visual failure.

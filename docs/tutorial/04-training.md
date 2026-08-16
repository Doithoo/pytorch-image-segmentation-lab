# Tutorial 04: From One Batch to a Resumable Run

[简体中文](04-training.zh-CN.md) | [Tutorial index](README.md)

The causal core is small:

```python
optimizer.zero_grad(set_to_none=True)
logits = model(images)
loss, components = criterion(logits, targets)
loss.backward()
optimizer.step()
```

Backward computes gradients; optimizer step changes parameters. The full
Trainer adds validation, records, and checkpoints around this sequence.

## Loss and dry run

Cross-entropy classifies valid pixels. Soft Dice emphasizes foreground overlap.
Both must filter index 255 consistently. The minimal config uses CE alone; the
reference config uses equal CE and Dice weights.

```bash
uv run segment train --config configs/learning_minimal.yaml --dry-run
```

Dry run loads one production batch and performs a real update. It prints device,
shapes, values, valid-pixel ratio, and loss, then `dry-run OK`. It does not create
a normal run or measure generalization.

## Read the run

```bash
uv run segment train --config configs/learning_minimal.yaml \
  --set run_name first-unet
uv run python scripts/plot_metrics.py artifacts/first-unet/metrics.csv \
  --output artifacts/first-unet/loss_curve.png
```

Training enables gradients and updates. Validation uses eval mode and no-grad.
`metrics.csv` records train/valid loss, pet IoU, and LR. Falling train loss with
rising validation loss suggests overfitting; falling loss with zero pet IoU can
indicate a background-only solution.

`last.pt` is the newest resumable state. `best.pt` is the highest validation pet
IoU state for evaluation. Checkpoints include model, optimizer, scheduler,
configuration, labels, manifest hashes, RNG state, version, and revision. Load
only trusted PyTorch checkpoints.

## Resume semantics

```bash
uv run segment train --config configs/reference_unet.yaml \
  --resume artifacts/reference-unet/last.pt
```

Resume rejects changes to model, preprocessing, loss, batch semantics, labels,
or manifests. With no scheduler, total epochs may be increased. A cosine
checkpoint must keep its planned total because saved `T_max` already defines the
schedule; start a new run for an extended cosine plan.

Seeds improve same-environment repeatability but do not erase device and library
differences or replace multi-seed evidence. Continue to
[evaluation and inference](05-evaluation-and-inference.md).

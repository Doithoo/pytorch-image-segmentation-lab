# Five Small Programs to Run First

[简体中文](README.zh-CN.md) | [Practice path](../docs/tutorial/learning-path.md)

Run these in numeric order, predict the output, then compare code. Each isolates
one production idea and links to its full module.

| Example | Observe | Production code |
| --- | --- | --- |
| 01 mask/logits | raw and mapped values; tensor shapes | dataset, losses |
| 02 paired transform | image and mask move left-to-right together | transforms |
| 03 DataLoader | image has channels; target does not | dataset |
| 04 training loop | loss components and a real parameter update | U-Net, Trainer |
| 05 checkpoint overlay | restored model writes raw/color/overlay PNG | Predictor |

```bash
uv run python examples/01_mask_and_logits.py
uv run python examples/02_paired_transform.py
uv run python examples/03_dataloader_batch.py
uv run python examples/04_minimal_training_loop.py
```

The first four need no downloaded data. Example 05 needs your checkpoint and an
image:

```bash
uv run python examples/05_checkpoint_overlay.py \
  --checkpoint artifacts/first-unet/best.pt \
  --image data/raw/images/Abyssinian_1.jpg \
  --output artifacts/example-prediction.png --device cpu
```

Use the `segment` CLI for complete experiments.

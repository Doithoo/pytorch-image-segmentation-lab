# Runnable Examples

[简体中文](README.zh-CN.md) | [Learning path](../docs/learning-path.md)

Run these files in numeric order. Each is intentionally small and names its
production counterpart in the module docstring.

## Examples 01-02: Pixel Meaning And Alignment

Example 01 maps Oxford labels, creates logits, calculates loss, and predicts
class indices. Example 02 forces a horizontal flip to demonstrate that image
and mask geometry must move together.

```bash
uv run python examples/01_mask_and_logits.py
uv run python examples/02_paired_transform.py
```

## Examples 03-04: Batch And Optimization

Example 03 shows `[B,C,H,W]` images beside `[B,H,W]` masks. Example 04 runs one
complete U-Net optimization step and exposes loss components.

```bash
uv run python examples/03_dataloader_batch.py
uv run python examples/04_minimal_training_loop.py
```

## Example 05: Checkpoint Inference

This example needs a checkpoint and image. It restores model construction and
preprocessing from the checkpoint and saves raw, color, and overlay outputs.

```bash
uv run python examples/05_checkpoint_overlay.py \
  --checkpoint artifacts/learning-unet/best.pt \
  --image data/raw/images/Abyssinian_1.jpg --device cpu
```

The examples are teaching surfaces, not alternative implementations. Use the
`segment` CLI for experiments.

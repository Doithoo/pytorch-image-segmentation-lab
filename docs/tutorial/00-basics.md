# Tutorial 00: Four Segmentation Tensors

[简体中文](00-basics.zh-CN.md) | [Tutorial index](README.md)

Separate images, targets, logits, and predictions before studying U-Net or
metrics:

```text
RGB batch       [B,3,H,W] float32
    -> model
logits          [B,2,H,W] float32
    -> argmax(1)
prediction      [B,H,W]   int64

target          [B,H,W]   int64
```

Classification produces scores per image. Segmentation produces scores per
pixel. Channel 0 scores background and channel 1 scores pet. The target has no
channel dimension because each pixel stores one class index.

## Run the shape example

```bash
uv run python examples/01_mask_and_logits.py
```

The example maps a 4x4 Oxford mask, creates deterministic logits, computes
cross-entropy, and predicts classes. Cross-entropy receives raw logits and an
integer target. Do not call `argmax` first: it discards score differences and
does not provide useful gradients.

## Logits and probabilities

Logits can be negative and do not sum to one. `cross_entropy` includes a stable
log-softmax, so training should pass logits directly. Apply softmax only when a
probability is needed for display or another algorithm. Argmax gives the same
class before and after softmax.

## Masks contain class IDs

Oxford values `1/2/3` mean pet/background/unclassified border and become
`1/0/255`. Value 255 is ignored by loss and metrics, not predicted as a class.
Resize images with bilinear interpolation and masks with nearest-neighbor;
bilinear mask resize invents invalid intermediate class values.

## Loss is not the report

Loss provides gradients. IoU and Dice measure overlap. Pixel accuracy can be
high for a model that predicts background everywhere, so the project selects
checkpoints with validation pet IoU.

For `B=4`, `H=W=128`, write all four shapes and dtypes, then verify them with:

```bash
uv run python examples/03_dataloader_batch.py
```

Continue to [data and masks](02-data-and-masks.md), or read
[environment setup](01-environment.md) if commands are not working yet.

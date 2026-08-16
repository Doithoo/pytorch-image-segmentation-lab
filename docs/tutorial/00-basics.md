# Tutorial 00: Segmentation Basics

[简体中文](00-basics.zh-CN.md) | [Tutorial index](../learning-path.md)

This chapter builds the tensor vocabulary used everywhere else.

## Classification Versus Segmentation

Image classification returns one class per image. Semantic segmentation returns
one class per pixel. A batch of RGB images is `[B,3,H,W]`; binary segmentation
logits are `[B,2,H,W]`; integer targets and predictions are `[B,H,W]`.

The two logits at one pixel are unnormalized scores for background and pet.
Cross-entropy applies softmax internally. Use `argmax(dim=1)` only when a class
decision is needed for metrics or output.

## Run The Shape Example

```bash
uv run python examples/01_mask_and_logits.py
```

Find the raw mask, mapped target, logits, and prediction in the file. Explain
why the target has no channel dimension and why logits must not be colorized
before the loss.

## Masks Are Data

A mask pixel is an integer category, not brightness. Resizing it with bilinear
interpolation could invent values such as 1.4, which are not classes. Masks use
nearest-neighbor interpolation and are stored separately from visual colors.

Oxford raw label 3 is unclassified and maps to 255. Ignoring it means the
project makes no claim about its class; it does not mean the model learns a
third class.

## Check Your Understanding

For `B=4`, `H=W=128`, write the image, target, logits, and prediction shapes.
Identify which tensors contain floating-point values. Predict what goes wrong
if you call `argmax` before cross-entropy.

Next: [Tutorial 01](01-environment.md).

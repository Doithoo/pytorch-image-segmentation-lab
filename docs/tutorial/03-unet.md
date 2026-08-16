# Tutorial 03: U-Net

[简体中文](03-unet.zh-CN.md) | [Tutorial index](../learning-path.md)

U-Net combines spatial context from an encoder with fine detail from skip
connections.

## Encoder

Each `ConvBlock` applies convolution, batch normalization, and ReLU twice. A
max-pool halves height and width while channels grow. Deeper features see more
context but lose exact boundary location.

## Bottleneck And Decoder

The bottleneck processes the smallest feature map. Each decoder block uses a
transposed convolution to double spatial size, concatenates the matching
encoder feature, and refines the result. Skip connections restore detail that
cannot be recovered from the bottleneck alone.

Input dimensions must be divisible by 16 because four pooling operations are
reversed exactly. The final `1x1` convolution maps features to two logits per
pixel without softmax.

## Run A Forward Pass

```bash
uv run python examples/04_minimal_training_loop.py
```

Open `src/image_segmenter/models/unet.py`. Trace the shapes after every encoder,
the bottleneck, every decoder, and the head for a `32x32` input. Increase
`base_channels` mentally and identify which dimensions change.

## Why Handwrite It

The primary U-Net is small enough to read and debug. The advanced DeepLabV3
adapter proves the trainer can accept another model with the same logits
contract, but it is not needed to learn the core workflow.

## Check Your Understanding

Explain why concatenation changes channels but not height, why masks never pass
through the model, and why applying softmax inside `forward` would duplicate
work expected by cross-entropy.

Next: [Tutorial 04](04-training.md).

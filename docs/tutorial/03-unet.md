# Tutorial 03: Read U-Net Through Shapes

[简体中文](03-unet.zh-CN.md) | [Tutorial index](README.md)

Follow `[B,3,128,128]` through the forward method before studying every layer:

```text
input       3 x 128 x 128
encoder    16 x 128 -> 32 x 64 -> 64 x 32 -> 128 x 16
bottleneck 256 x 8
decoder    128 x 16 -> 64 x 32 -> 32 x 64 -> 16 x 128
logits      2 x 128 x 128
```

The encoder trades spatial detail for context. The decoder upsamples deep
features and concatenates same-resolution encoder features. Skip connections
restore location detail that the bottleneck cannot reconstruct alone.

## Read the implementation

Open `src/image_segmenter/models/unet.py` and map each `forward` variable to the
shape path. `ConvBlock` preserves height/width, max-pool halves it, and
`torch.cat(..., dim=1)` increases channels without changing spatial size.

Four pooling stages require dimensions divisible by 16. Configuration validates
that early instead of failing at a decoder concatenation.

The final 1x1 convolution maps features to two per-pixel logits. It does not
apply softmax; cross-entropy needs logits and inference uses argmax.

```bash
uv run python examples/04_minimal_training_loop.py
uv run python -c "import torch; from image_segmenter.models.unet import SmallUNet; m=SmallUNet(base_channels=4); print(m(torch.rand(2,3,32,32)).shape)"
```

Doubling `base_channels` raises parameters, memory, and compute by much more than
two. Compare widths with the same data, loss, epochs, and seed.

Torchvision DeepLabV3 returns a dictionary, so `DeepLabV3Adapter` converts it to
the same direct logits contract. The rest of the pipeline remains model-
agnostic. Continue to [training](04-training.md).

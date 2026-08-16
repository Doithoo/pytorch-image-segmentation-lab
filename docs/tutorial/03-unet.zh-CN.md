# 教程 03：U-Net

[English](03-unet.md) | [教程索引](../learning-path.zh-CN.md)

U-Net 将编码器的空间上下文与 skip connection 的细节结合起来。

## 编码器

每个 `ConvBlock` 两次应用卷积、batch normalization 和 ReLU。Max-pool 把高宽
减半，同时增加通道。更深特征看到更多上下文，但会损失精确边界位置。

## Bottleneck 与解码器

Bottleneck 处理最小特征图。每个解码块使用转置卷积把空间尺寸加倍，拼接对应编码
特征，再细化结果。Skip connection 恢复仅靠 bottleneck 无法找回的细节。

输入尺寸必须能被 16 整除，因为四次池化需要精确反转。最终 `1x1` 卷积把特征映射
为每像素两个 logits，不应用 softmax。

## 运行前向传播

```bash
uv run python examples/04_minimal_training_loop.py
```

打开 `src/image_segmenter/models/unet.py`。对 `32x32` 输入追踪每个编码器、
bottleneck、解码器和 head 的形状。想象增加 `base_channels`，指出哪些维度改变。

## 为什么手写

主 U-Net 足够小，可以完整阅读和调试。进阶 DeepLabV3 适配器证明 trainer 能接收
遵守相同 logits 契约的另一模型，但学习核心流程不需要它。

## 检查理解

解释为什么拼接改变通道但不改变高度，为什么掩码从不经过模型，以及在 `forward`
内应用 softmax 为何会重复交叉熵预期的工作。

下一章：[教程 04](04-training.zh-CN.md)。

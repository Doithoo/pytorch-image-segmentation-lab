# 教程 03：沿着形状读一遍 U-Net

[English](03-unet.md) | [教程索引](README.zh-CN.md)

第一次读分割模型，不要从论文图或每个卷积参数开始。拿一张 `[B,3,128,128]` 图片，
沿 forward 追踪空间尺寸和通道数，先回答“信息怎样下去，又怎样回来”。

## 为什么分类模型的输出不够

普通分类模型最后把空间信息压成一个向量，输出一张图的类别。分割需要为每个像素输出
类别，因此既要理解“这是宠物”，又要保留“宠物边缘在哪里”。

U-Net 用两条路径完成这件事：

```text
encoder:  分辨率逐步减半，通道增加，看到更大范围的上下文
decoder:  分辨率逐步恢复，结合 encoder 的细节定位边缘
```

## 先看完整形状

默认 `base_channels=16`、输入 128 时，主要形状是：

```text
input             [B,   3, 128, 128]
enc1              [B,  16, 128, 128]
enc2              [B,  32,  64,  64]
enc3              [B,  64,  32,  32]
enc4              [B, 128,  16,  16]
bottleneck         [B, 256,   8,   8]
dec4               [B, 128,  16,  16]
dec3               [B,  64,  32,  32]
dec2               [B,  32,  64,  64]
dec1               [B,  16, 128, 128]
head logits        [B,   2, 128, 128]
```

打开 [unet.py](../../src/image_segmenter/models/unet.py)，先只找 `SmallUNet.forward`，
把上表中的每一行对应到一个变量。

## ConvBlock 做了什么

每个 `ConvBlock` 使用两次 `3x3 convolution -> batch normalization -> ReLU`。padding
保持高宽不变，卷积更新局部特征，BatchNorm 稳定每个通道的统计，ReLU 提供非线性。

第一遍不必推导参数量。先确认：同一个 block 内高宽不变，通道数由构造参数决定。

## 下采样得到上下文，也丢掉位置细节

MaxPool 每次把高宽减半。更深层的一个位置对应原图更大区域，因此能判断局部纹理属于
猫、狗还是背景；代价是细腿、耳尖和毛发边界的位置变得粗糙。

四次池化要求输入高宽能被 16 整除。配置加载时提前验证这一点，避免进入 decoder 才
出现无法拼接的形状错误。

## skip connection 为什么重要

decoder 先用转置卷积放大特征，再与 encoder 同尺寸的特征拼接：

```python
x = self.up(x)
x = torch.cat([x, skip], dim=1)
x = self.conv(x)
```

`torch.cat(..., dim=1)` 增加通道数，不改变高宽。深层特征提供“是什么”，浅层 skip
提供“边缘在哪里”。随后 ConvBlock 学习怎样融合两者。

如果只靠 bottleneck 上采样，模型仍可画出大致区域，但细节通常更差。这也是 U-Net 在
医学图像和小数据分割中经典的原因之一。

## 最后一层为什么是 1x1 卷积

`1x1` 卷积在每个像素位置把 16 个特征通道映射为两个类别 logits。它不负责扩大感受野，
而是完成“特征 -> 类别分数”的最后投影。

head 不使用 softmax。训练 loss 需要原始 logits；推理时再用 `argmax(dim=1)` 得到
0/1 类别索引。

## 运行一次真实 forward

```bash
uv run python examples/04_minimal_training_loop.py
```

示例创建一个更窄的 U-Net，打印 logits、交叉熵、Dice 和总 loss。然后可以在本地临时
给 `forward` 加 shape print，理解后撤销；不要把调试 print 留在训练循环里。

也可以直接运行：

```bash
uv run python -c "import torch; from image_segmenter.models.unet import SmallUNet; m=SmallUNet(base_channels=4); print(m(torch.rand(2,3,32,32)).shape)"
```

预期输出是 `torch.Size([2, 2, 32, 32])`：模型保持空间尺寸，只把输入 RGB 通道变成
每像素两个类别通道。

## 宽度改变了什么

`base_channels` 从 16 增到 32 时，每层通道大致翻倍，参数量和内存增长远超过两倍。
更宽不保证更好：小数据可能更快过拟合，CPU/MPS 也会明显变慢。

做宽度对比时保持输入、manifest、loss、epoch 和 seed 不变，并记录运行时间。不要同时
把 U-Net 换成 DeepLabV3、打开预训练权重又改变 loss，那样无法判断差异来自哪里。

## DeepLabV3 为什么通过适配器接入

torchvision DeepLabV3 返回字典，而 Trainer 期望直接得到 `[B,2,H,W]` logits。
`DeepLabV3Adapter` 在模型边界提取输出并恢复输入尺寸，使 Dataset、loss、Trainer 和
evaluator 不需要知道具体模型。

这是一种有用的接口约定，但不必在第一次学习 U-Net 时深入 torchvision 内部。

## 读完后回答

1. 四次池化后，128 为什么变成 8？
2. skip 拼接为什么改变通道但不改变高宽？
3. 模型为什么从不读取目标掩码？
4. 最后一层为什么输出两个通道却不使用 softmax？
5. 把 `base_channels` 翻倍会影响哪些张量和资源？

接着进入[教程 04：训练](04-training.zh-CN.md)，把 forward 放进真实优化循环。

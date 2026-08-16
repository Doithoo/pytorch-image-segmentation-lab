# 教程 00：先看懂分割里的四种张量

[English](00-basics.md) | [教程索引](README.zh-CN.md)

这一章只解决一个问题：图片、目标、logits 和预测分别长什么样。把这四个对象分清，
后面的 U-Net、loss 和指标就不再是一串孤立术语。

## 从一次预测开始

```text
RGB 图片批次                  [B, 3, H, W]  float32
        ↓ model(images)
每像素两个 logits             [B, 2, H, W]  float32
        ↓ argmax(dim=1)
预测类别掩码                  [B, H, W]     int64

训练目标掩码                  [B, H, W]     int64
```

分类模型通常为一张图输出一组类别分数；分割模型为每个像素输出一组分数。在本项目中，
通道 0 是背景分数，通道 1 是宠物分数。

`B` 是 batch size，`H/W` 是网络输入的高宽。目标没有颜色通道：每个位置只保存一个
整数类别，例如 0、1 或忽略索引 255。

## 运行最小示例

```bash
uv run python examples/01_mask_and_logits.py
```

示例先构造一个 `4x4` Oxford 风格掩码，再完成标签映射、随机 logits、交叉熵和类别
预测。注意两件事：

1. 目标从 `[H,W]` 增加 batch 维后交给交叉熵，没有 one-hot 通道。
2. `argmax` 只用于生成预测，loss 读取的是两个通道的原始 logits。

为什么不能先 `argmax`？因为 argmax 只留下“谁最大”，丢掉分数差距，而且几乎处处
不可导。模型就无法知道应该把正确类别提高多少。

## logits 不是概率

假设某个像素的 logits 是：

```text
background:  1.2
pet:         2.7
```

它们可以为负，也不要求总和为 1。softmax 可以把它们转换为概率，但 PyTorch 的
`cross_entropy` 已经在内部完成数值稳定的 log-softmax，因此训练代码应直接传 logits。

只有展示置信度或需要概率时才显式调用 softmax。生成最终类别时使用 argmax 即可；
softmax 不改变最大值所在的位置。

## 掩码不是灰度照片

掩码看起来像黑白或灰度图，但像素值表示类别，不表示亮度。Oxford trimap 使用：

```text
1 = pet
2 = background
3 = unclassified border
```

项目映射为：

```text
1 = pet
0 = background
255 = ignore
```

如果对掩码做双线性缩放，类别 1 和 2 之间可能出现 1.4、1.7 等值。对照片这是平滑，
对类别 ID 则是损坏。因此图片使用双线性插值，掩码使用最近邻插值。

## loss 和指标回答不同问题

交叉熵告诉优化器怎样调整参数；IoU 和 Dice 告诉人预测区域与目标区域重合得怎样。
训练可以组合交叉熵和 soft Dice，但选择 checkpoint 使用验证集 pet IoU。

不要把 pixel accuracy 当作唯一结论。若一张图 80% 是背景，一个总猜背景的模型也有
80% accuracy，但宠物 IoU 是 0。

## 自己检查一次形状

当 `B=4`、`H=W=128` 时，先写下答案再展开：

- 图片：`[4, 3, 128, 128]`，浮点值。
- 目标：`[4, 128, 128]`，整数类别。
- logits：`[4, 2, 128, 128]`，浮点值。
- 预测：`[4, 128, 128]`，整数类别。

再运行：

```bash
uv run python examples/03_dataloader_batch.py
```

把输出与答案对照。如果你能解释目标为何没有通道维、loss 为何不先 argmax，以及掩码
为何不能双线性缩放，就可以进入[教程 02：数据与掩码](02-data-and-masks.zh-CN.md)。

环境还没有跑通时先看[教程 01](01-environment.zh-CN.md)。

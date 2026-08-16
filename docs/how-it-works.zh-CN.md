# 系统如何工作

[English](how-it-works.md) | [返回 README](../README.zh-CN.md)

语义分割为每个像素分配一个类别。本项目明确展示完整路径，使学习者可以检查每一种
表示。

## 数据流

```text
official archives -> images/masks -> manifests -> paired transform
-> image [B,3,H,W] + target [B,H,W] -> U-Net logits [B,2,H,W]
-> loss/optimizer or argmax -> metrics + mask artifacts
```

清单把数据准备与训练解耦。路径可以是相对路径，因此准备后的数据集可以整体移动。
图像和掩码的几何始终对齐；颜色归一化只应用于图像。

## 标签流

Oxford 原始值为 `1=宠物`、`2=背景`、`3=未分类`。Dataset 将其映射为 `1`、`0`
和 `255`。交叉熵把 255 作为 `ignore_index`，Dice 和指标也显式过滤相同像素。

模型永远不预测 255。两个通道分别表示背景和宠物。原始 trimap、映射目标、logits、
预测类别掩码、彩色渲染和透明叠加图是用途不同的对象。

## 学习流

训练时，U-Net 输出 logits。交叉熵鼓励每个有效像素的正确类别，前景 soft Dice
强调重叠，降低大面积背景的影响。组合损失算完后，梯度才更新权重。

Trainer 每轮测量验证集宠物 IoU。`last.pt` 表示最近训练状态，`best.pt` 表示验证
宠物 IoU 最高的状态，两者可能不同。

## 评估流

官方测试集在最终评估前保持不动。预测和目标共同更新一个数据集级混淆矩阵。逐图
指标也会写出以便诊断，但它们的算术均值与全局 IoU 不是同一个统计量。

当某图的目标和预测均无前景时，逐图宠物 IoU 和 Dice 没有定义。评估器会计数并
从逐图均值排除。可视化叠加图帮助解释数值分数来自哪里。

## 复现流

最终配置、来源元数据、清单哈希、运行元数据、每轮指标和 checkpoint 状态一起保存。
推理从 checkpoint 恢复输入尺寸和归一化，避免 CLI 默认值变化暗中影响旧模型。

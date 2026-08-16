# 教程 05：评估与推理

[English](05-evaluation-and-inference.md) | [教程索引](../learning-path.zh-CN.md)

评估衡量有标签划分；推理为无标签图像生成掩码。两者都从 checkpoint 恢复模型和
预处理。

## 评估选定 Checkpoint

```bash
uv run segment evaluate --checkpoint artifacts/learning-unet/best.pt \
  --split test --plot --device cpu
```

开发时使用验证集，只对最终选定运行使用官方测试集。评估器写出全局指标、逐类别行、
逐图行、代表性对比、最差案例和可选损失曲线。

## 正确阅读指标

宠物 IoU 是交集除以并集，宠物 Dice 是两倍交集除以预测与目标宠物像素之和。
Mean IoU 包含背景；像素准确率可能被背景主导。验证宠物 IoU 用于选择 checkpoint。

数据集级 IoU 来自一个累积混淆矩阵。逐图 IoU 均值给予每个合格图像相同权重。目标
和预测均为空时，逐图宠物 IoU 无定义，因此样本被计数并排除。

## 可视化诊断

对比网格展示图像、目标、预测和错误。找到假阳性、假阴性、真阳性和忽略像素。检查
错误是否集中在毛发、细腿、异常姿态、小宠物或混淆背景。曲线说明何时变化，叠加图
说明如何变化。

## 预测新图像

```bash
uv run segment predict --checkpoint artifacts/learning-unet/best.pt \
  --image data/raw/images/Abyssinian_1.jpg \
  --output artifacts/predictions/Abyssinian_1.png --device cpu
```

原始输出 PNG 包含类别索引 0 和 1；`.color.png` 供人查看；`.overlay.png` 把宠物
像素混合到原图。绝不能把彩色图或叠加图重新作为目标掩码。

## 继续学习

按照[实验指南](../experiments.zh-CN.md)写一个假设，或通过
[使用自己的数据](../using-your-data.zh-CN.md)适配二类配对数据。保持官方测试集隔离，
并为每个实测结果报告限制。

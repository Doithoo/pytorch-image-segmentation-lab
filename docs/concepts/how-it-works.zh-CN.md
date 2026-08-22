# 一张样本怎样走完整个系统

[English](how-it-works.md) | [文档导航](../README.zh-CN.md)

本文按数据流解释“做了什么”和“为什么这样做”。第一次操作请跟随
[实践路线](../tutorial/learning-path.zh-CN.md)，需要理解设计原因时再回来查阅。

## 总览

```text
官方归档/配对目录
  -> 校验并写 manifest
  -> Dataset 读取 image + raw mask
  -> 成对 resize/flip + 标签映射
  -> image [B,3,H,W] + target [B,H,W]
  -> U-Net logits [B,2,H,W]
  -> loss/backward/update 或 argmax
  -> checkpoint、指标表、掩码和叠加图
```

## manifest 把“数据是谁”固定下来

训练代码不遍历目录，而是读取 CSV。固定成员和相对路径让同一 manifest 可以在不同
模型实验间复用，也可以随数据目录一起移动。SHA-256 用于确认两次运行是否使用相同
成员，不表示数据本身一定没有语义问题。

Oxford 官方 test 不重新随机切分；trainval 按品种切成 train/valid。通用数据按配置
比例随机切分，并拒绝空 split。

## Dataset 统一标签语义

原始 Oxford `1/2/3` 映射为 `1/0/255`。模型只预测 0 和 1，loss 与指标都跳过 255。
映射发生在 Dataset，而不是散落在 loss、评测和绘图中，因此所有消费者看到同一种目标。

图片和掩码共享随机几何变换。颜色归一化只应用于图片；mask 保持 int64 类别 ID。

## U-Net 只承诺输出 logits

模型不知道 manifest、原始标签值或 checkpoint 目录。它把 `[B,3,H,W]` 转成
`[B,2,H,W]`。手写 U-Net 与 DeepLab adapter 遵守同一输入/输出形状，Trainer 不需要
按模型名称写分支。

## loss 提供梯度，指标提供解释

交叉熵逐像素学习类别；soft Dice 强调前景重叠。两者可以组合，但训练 loss 不是最终
报告。验证 pet IoU 选择 best checkpoint，因为 pixel accuracy 容易被背景占比主导。

`SegmentationMetrics` 累积混淆矩阵，不保存整个数据集预测。全局 IoU 从总计数计算；
逐图指标用于发现某类图片是否特别差。两者不能互相替换。

## checkpoint 让推理不依赖记忆

checkpoint 保存模型、预处理、标签、配置和版本。Predictor 不要求用户再次输入网络尺寸
或归一化参数，避免训练与推理漂移。

训练恢复还需要 optimizer、scheduler、epoch、最佳分数和随机状态。恢复前核对 manifest
与训练语义，防止把不同实验拼接到同一个目录。

## 数值产物与视觉产物各司其职

JSON/CSV 适合比较和自动处理；原始 0/1 PNG 适合其他程序；color/overlay 适合人看。
项目分别保存它们，避免展示颜色污染类别数据。

评测同时保留代表样本与最差案例。汇总指标说明“总体有多好”，叠加图说明“错在哪里”。
可信报告需要两者。

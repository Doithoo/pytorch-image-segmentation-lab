# 教程 04：训练

[English](04-training.md) | [教程索引](../learning-path.zh-CN.md)

训练重复一个小序列，同时保存足够状态以解释和恢复过程。

## 一个批次

```text
load -> forward -> loss -> zero gradients -> backward -> optional clip -> step
```

交叉熵使用有效背景和宠物像素。前景 soft Dice 使用宠物概率并过滤忽略像素。如果
某批次没有有效前景，Dice 项会停用，而不是奖励空掩码。

## 先运行 Dry Run

```bash
uv run segment train --config configs/learning_minimal.yaml --dry-run
```

Dry run 执行真实优化器更新，并报告张量形状、映射掩码值、有效像素比例、损失分量
和设备，但不创建常规运行。投入数小时训练前先修复这里的所有问题。

## Epoch 与验证

```bash
uv run segment train --config configs/learning_minimal.yaml
```

训练批次更新权重；验证批次使用 eval 模式且不计算梯度。每轮后，指标追加到
`metrics.csv`，`last.pt` 更新，只有验证宠物 IoU 改善时才更新 `best.pt`。

训练损失可能下降，而验证 IoU 停滞或下降，因此最新 checkpoint 不一定是泛化最好
的 checkpoint。

## 精确恢复

恢复会还原模型、优化器、调度器、epoch、最佳分数和随机状态，包括 shuffle 序列。
模型设置、预处理、损失与训练语义、标签、忽略索引或清单哈希不匹配时会拒绝恢复。
数据路径可以移动，总 `train.epochs` 可以增加；不要指向无关运行。

## 阅读产物

`config.yaml` 是最终配置，`run.yaml` 记录环境和来源事实，`metrics.csv` 是只追加
的 epoch 证据。Checkpoint 是受信任项目产物并使用 Python 序列化；不要加载未知
来源文件。

下一章：[教程 05](05-evaluation-and-inference.zh-CN.md)。

# 实验

[English](experiments.md) | [返回 README](../README.zh-CN.md)

一个实验回答一个写明的问题。保持数据成员、随机种子、评估协议和所有无关字段固定。

## 建立基线

先原样运行学习配置。保存最终配置、清单哈希、指标和最佳 checkpoint。失败或缓慢的
基线也是有用信息；不要通过同时调整多个参数来掩盖它。

```bash
uv run segment show-config --config configs/learning_minimal.yaml
uv run segment train --config configs/learning_minimal.yaml
```

## 只改变一个变量

聚焦问题示例：

- 与仅交叉熵相比，前景 Dice 是否改善宠物 IoU？
- 更宽的 U-Net 带来的收益是否值得时间和内存？
- 水平翻转是否改善验证表现？
- 在清楚记录预训练权重条件时，手写 U-Net 与 DeepLabV3 表现如何？

为每次运行设置不同名称。使用 `--set` 做紧凑的单变量变化，然后检查打印的最终
配置。

```bash
uv run segment train --config configs/learning_minimal.yaml \
  --set loss.cross_entropy_weight 0.5 \
  --set loss.dice_weight 0.5 --set run_name ce-dice
```

## 比较证据

用验证集宠物 IoU 选模型，用训练/验证曲线观察优化行为，用最差案例叠加图分析错误
特征。记录运行时间和设备。不要为每个尝试设置报告测试结果，否则测试集就变成了
验证集。

全局宠物 IoU 和逐图宠物 IoU 均值回答不同问题，必须写明统计量。当目标与预测均
无前景时，该图会从逐图均值排除。

## 避免无效对照

如果划分成员、隐藏预处理、样本上限或预训练初始化不同，就不能声称只比较了指定
模型。不要因为测试分数最高而选择结果。只有一个随机种子时，不要隐瞒这一限制而
推导一般结论。

报告格式参考[参考运行](reference-run.zh-CN.md)，覆盖审计参考
[配置流](configuration-flow.zh-CN.md)。

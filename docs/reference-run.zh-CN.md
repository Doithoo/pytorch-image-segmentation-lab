# 参考运行

[English](reference-run.md) | [返回 README](../README.zh-CN.md)

本页记录 `configs/reference_unet.yaml` 的实测证据。只有完整运行并在未改动的官方
测试集完成评估、产物提交后，才填写数值。

## 协议

- 数据集：使用固定 SHA-256 的 Oxford-IIIT Pet 官方归档。
- 划分：官方测试集不变；trainval 按配置种子在每个品种内进行 80/20 划分。
- 标签：原始 1 为宠物、2 为背景、3 为忽略索引 255。
- 选择：25 轮中验证集宠物前景 IoU 最高的 checkpoint。
- 测试使用：只对选定 checkpoint 进行一次最终评估。

```bash
uv run segment train --config configs/reference_unet.yaml
uv run segment evaluate --checkpoint artifacts/reference-unet/best.pt \
  --split test --plot
```

## 环境

本次运行使用 macOS 26.5.2、Apple arm64 与 MPS，Python 3.11.15、PyTorch
2.13.0、torchvision 0.28.0。训练代码 revision 为
`1deff1a03ba2d4549c538b4eef36df3868abe92b`。

训练开始于 `2026-08-16T04:35:04Z`，结束于 `2026-08-16T09:55:57Z`。
端到端耗时 19,253 秒（5:20:53），包含一次重启。从第 15 轮恢复到第 25 轮的
进程耗时 7,910.98 秒；最终测试评估耗时 122.98 秒。

第一个 MPS 进程完成第 1-14 轮后，在第 15 轮开始前触发 Metal 高水位限制。
进程退出时产物未损坏，并从原子写入的第 14 轮 `last.pt` 恢复。安全限制没有被
关闭。该设备限制可能改变运行时间。记录 revision 恢复了模型、优化器、调度器和
全局随机状态，但其独立 DataLoader shuffle generator 未写入 checkpoint，因此恢复
进程从头开始了该 shuffle 序列。运行后的回归修复移除了这项未保存状态。本次实测
结果仍绑定记录 revision，不应视为与未中断运行逐比特等价。

## 实测结果

最佳 checkpoint 来自第 24 轮，验证集宠物 IoU 为 `0.902836`。它在全部 3,669
个官方测试样本上仅评估一次，并超过测试集宠物 IoU 至少为 0.70 的要求。

| 指标 | 实测值 |
| --- | ---: |
| 测试集宠物 IoU（全局） | 0.906669 |
| 测试集宠物 Dice（全局） | 0.951050 |
| 测试集背景 IoU（全局） | 0.950423 |
| 测试集 mean IoU（全局） | 0.928546 |
| 测试集像素准确率 | 0.966539 |
| 测试集宠物 IoU（逐图均值） | 0.905472 |
| 有效像素 | 210,899,866 |
| 忽略像素 | 29,551,718 |
| 测试样本 | 3,669 |
| 空目标样本 | 7 |

已提交证据包括[最终配置](../artifacts/reference-unet/config.yaml)、
[运行元数据](../artifacts/reference-unet/run.yaml)、[25 轮指标](../artifacts/reference-unet/metrics.csv)、
[最佳 checkpoint](../artifacts/reference-unet/best.pt)、[评估摘要](../artifacts/reference-unet/evaluation/metrics.json)、
[逐类表格](../artifacts/reference-unet/evaluation/per_class_metrics.csv)、
[逐图表格](../artifacts/reference-unet/evaluation/per_image_metrics.csv)、
[损失曲线](../artifacts/reference-unet/evaluation/loss_curve.png)和
[数据预览](../artifacts/reference-unet/dataset_preview.png)。有界的
[样例叠加图](../artifacts/reference-unet/evaluation/sample_overlays)与
[最差案例](../artifacts/reference-unet/evaluation/worst_cases)用于视觉检查。

## 解释

结果证明本仓库流水线能在记录条件下学到有效宠物掩码。它不代表 SOTA，不保证
不同硬件完全相同，也不验证其他数据集。

分数必须与损失曲线、代表性叠加图和最差案例一起检查。背景泄漏、细小结构漏检和
忽略 trimap 区域会揭示单一汇总数字隐藏的限制。七个官方测试 mask 只包含原始值
2，因此没有目标前景。视觉检查还在最低逐图 IoU 案例中发现了反转或空间不一致的
上游 trimap。这些样本保持原样，并包含在所有报告的聚合指标中。

## 复现或审计

使用 [source.yaml](../data/manifests/source.yaml) 及其清单 SHA-256 确认划分身份：
train 为 `f7e7af...737171`、valid 为 `872ed4...a5897d`、test 为
`619642...f757c`。完整产物哈希记录在 `run.yaml` 中。比较保存的最终
`config.yaml`，不要只相信配置文件名。检查 checkpoint 元数据中的包和 Git
revision。改变变量前阅读[实验指南](experiments.zh-CN.md)。

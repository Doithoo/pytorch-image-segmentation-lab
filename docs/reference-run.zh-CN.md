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

状态：等待实测参考运行。发布记录将包含操作系统、处理器或加速器、Python、PyTorch、
torchvision、包 revision、运行时间和设备限制。

## 实测结果

状态：等待实测参考运行。在提交 `metrics.json` 和清单校验和之前不声称任何分数。
发布验收要求测试集宠物 IoU 至少为 0.70。

最终表格会分别列出全局宠物 IoU、宠物 Dice、mean IoU、像素准确率、有效/忽略
像素、样本数和逐图宠物 IoU 均值。

## 解释

结果将证明本仓库流水线能在记录条件下学到有效宠物掩码。它不代表 SOTA，不保证
不同硬件完全相同，也不验证其他数据集。

分数必须与损失曲线、代表性叠加图和最差案例一起检查。背景泄漏、细小结构漏检和
忽略 trimap 区域可能揭示单一汇总数字隐藏的限制。

## 复现或审计

使用 `source.yaml` 和清单 SHA-256 确认划分身份。比较保存的最终 `config.yaml`，
不要只相信配置文件名。检查 checkpoint 元数据中的包和 Git revision。改变变量前
阅读[实验指南](experiments.zh-CN.md)。

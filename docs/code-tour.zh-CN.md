# 代码导览

[English](code-tour.md) | [返回 README](../README.zh-CN.md)

包结构有意保持小巧。建议沿着一个样本从清单行走到预测结果，不要按字母顺序阅读
文件。

## 数据边界

`data/manifest.py` 校验配对的图像和掩码并写出可移植 CSV。Oxford 逻辑位于这里和
`scripts/download_data.py`，Dataset 不需要知道归档结构。`data/transforms.py`
负责成对几何变换，`data/dataset.py` 加载一行并把原始标签映射为训练索引。

不变量：掩码始终是整数类别 ID，原始标签 3 在进入损失或指标前变为忽略索引 255。

## 模型与损失边界

`models/unet.py` 是清晰可读的主模型。四级编码特征成为 skip connection；解码块
上采样并拼接它们。注册表以同一 logits 契约返回 U-Net 或进阶 DeepLab 适配器。

`training/losses.py` 组合交叉熵和前景 soft Dice，同时排除忽略像素。它单独返回
各损失分量，让日志清楚显示优化器在最小化什么。

## 训练边界

`training/train.py` 组装配置、DataLoader、模型、优化器、调度器和设备。
`training/trainer.py` 管理 epoch 循环，并用验证集宠物 IoU 选择 `best.pt`。
`training/checkpoint.py` 通过 schema 校验原子保存模型及恢复状态。

```bash
uv run segment train --config configs/learning_minimal.yaml --dry-run
```

阅读这三个模块时运行该命令：它会跨过每个边界一次，不依赖 notebook 的隐藏状态。

## 评估与推理边界

`evaluation/metrics.py` 在有效像素上累积混淆矩阵。`evaluation/evaluate.py` 恢复
checkpoint、写表格、排列最差案例并绘制对比。`evaluation/visualization.py` 将
原始掩码与面向人类的彩色图严格分开。

`inference/predictor.py` 从 checkpoint 恢复模型和预处理，在网络尺寸预测后用
最近邻插值把类别索引还原到原图尺寸。

## CLI 边界

`cli.py` 解析命令，并把预期错误变成简洁消息。耗时工作前会完成配置解析。下一步
阅读[配置流](configuration-flow.zh-CN.md)，再用五个
[示例](../examples/README.zh-CN.md)作为聚焦入口。

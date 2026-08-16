# 配置文件

[English](README.md) | [配置参考](../docs/config-reference.zh-CN.md)

这些 YAML 文件描述可复现实验。训练前始终检查最终解析配置。

## 最小学习配置

`learning_minimal.yaml` 是第一个运行：CPU 友好的样本上限、128 像素输入、窄
U-Net、两轮和交叉熵。它用于证明流程，其分数不是参考结果。

```bash
uv run segment show-config --config configs/learning_minimal.yaml
uv run segment train --config configs/learning_minimal.yaml --dry-run
```

## 参考 U-Net

`reference_unet.yaml` 是固定的 25 轮实测运行，使用完整的已准备训练/验证划分以及
交叉熵加前景 Dice。比较想法时不要原地修改；复制文件或使用带名称的覆盖。

## DeepLabV3 对照

`deeplabv3_resnet50.yaml` 是进阶架构对照，其预训练设置可能下载 torchvision
权重。必须记录是否使用权重；预训练模型与随机初始化 U-Net 不是纯架构对照。

## 安全覆盖

只用 `--set` 改一个明确变量，并设置不同 `run_name`。图像尺寸必须能被 16 整除。
模型类别、原始标签、忽略索引和预处理都会成为 checkpoint 兼容性数据。

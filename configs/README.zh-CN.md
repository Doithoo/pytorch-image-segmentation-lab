# 可直接运行的配置

[English](README.md) | [配置参考](../docs/reference/config-reference.zh-CN.md)

不要从空白 YAML 开始。先选择最接近目标的一份，运行 `show-config`，再用一个有名字的
覆盖做实验。

| 配置 | 用途 | 主要特点 |
| --- | --- | --- |
| `learning_minimal.yaml` | 第一次跑通 | 128 px、最多 256/64/64 样本、两轮、CE、小 U-Net |
| `reference_unet.yaml` | 完整手写模型实验 | 256 px、25 轮、base 32、CE+Dice、cosine |
| `deeplabv3_resnet50.yaml` | 进阶方案对照 | 预训练 DeepLabV3、batch 4、15 轮 |
| `fcn_resnet50.yaml` / `lraspp_mobilenet.yaml` | torchvision 起点 | 不下载权重的模型切换 |
| `smp_unetplusplus.yaml` | 可选模型 | 需要 `model-zoo` extra，encoder 权重为 null |
| `custom_model_example.yaml` / `custom_dataset_example.yaml` | 扩展边界 | 外部 model/Dataset 工厂 |

```bash
uv run segment show-config --config configs/learning_minimal.yaml
uv run segment train --config configs/learning_minimal.yaml --dry-run
```

比较实验时不要原地修改参考配置。使用 `--set` 和不同 `run_name`，运行目录中的
`config.yaml` 会保存最终值。

DeepLab 的预训练权重可能联网下载。它与随机初始化 U-Net 的默认对比包含初始化、模型、
batch 和学习率等多项差异，应称为完整方案对比，而不是纯架构消融。

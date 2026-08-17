# 可用模型

[English](model-zoo.md) | [切换模型](../guides/using-models.zh-CN.md) | [文档导航](../README.zh-CN.md)

先运行 `uv run segment list-models` 查看当前环境，再用 `model-info 模型名` 查看参数、
输入限制和依赖状态。注册名是配置里的稳定标识，不代表模型性能排序。

| 名称 | 来源 | 额外依赖 | 使用时先注意 |
| --- | --- | --- | --- |
| `unet_small` | 本项目 | 无 | 结构便于阅读，`base_channels` 控制宽度 |
| `fcn_resnet50` | torchvision | 无 | 只接收三通道输入，骨干较大 |
| `deeplabv3_resnet50` | torchvision | 无 | 空洞卷积分割基线 |
| `deeplabv3_mobilenet_v3_large` | torchvision | 无 | MobileNet 骨干 |
| `lraspp_mobilenet_v3_large` | torchvision | 无 | 较轻的 torchvision 选项 |
| `smp_unet` | SMP | `model-zoo` | 可更换 encoder |
| `smp_unetplusplus` | SMP | `model-zoo` | decoder 连接比 U-Net 更密 |
| `smp_fpn` | SMP | `model-zoo` | 特征金字塔 decoder |
| `smp_pan` | SMP | `model-zoo` | 默认路径输入至少为 128×128 |

SMP 是可选依赖：

```bash
uv sync --extra dev --extra model-zoo
uv run segment model-info smp_unetplusplus
```

`pretrained: true` 可能下载 encoder 权重。离线练习或控制变量时，使用
`pretrained: false`，SMP 同时写 `encoder_weights: null`。仓库只把
[U-Net 参考运行](../recorded-run/README.zh-CN.md)作为实测结果，不据此推断其他模型的效果。

# Model zoo

[中文](model-zoo.zh-CN.md) | [Using models](using-models.md)

Use `uv run segment list-models` for the installed environment and `model-info NAME`
for parameters, constraints, and availability. Registered names are stable configuration
values; they are not a performance ranking.

| Name | Provider | Extra | Practical note |
| --- | --- | --- | --- |
| `unet_small` | this project | none | Readable baseline; `base_channels` controls width |
| `fcn_resnet50` | torchvision | none | Three-channel input; comparatively large backbone |
| `deeplabv3_resnet50` | torchvision | none | Atrous segmentation baseline |
| `deeplabv3_mobilenet_v3_large` | torchvision | none | MobileNet backbone |
| `lraspp_mobilenet_v3_large` | torchvision | none | Lighter torchvision option |
| `smp_unet` | SMP | `model-zoo` | Configurable encoder |
| `smp_unetplusplus` | SMP | `model-zoo` | Denser decoder than U-Net |
| `smp_fpn` | SMP | `model-zoo` | Feature pyramid decoder |
| `smp_pan` | SMP | `model-zoo` | Default path needs at least 128x128 input |

Install optional SMP models with:

```bash
uv sync --extra dev --extra model-zoo
uv run segment model-info smp_unetplusplus
```

`pretrained: true` can download encoder weights. For an offline or controlled comparison,
set `pretrained: false` and, for SMP, `encoder_weights: null`. Only the recorded U-Net run
in [reference-run.md](reference-run.md) is a measured result in this repository.

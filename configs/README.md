# Configuration Files

[简体中文](README.zh-CN.md) | [Configuration reference](../docs/config-reference.md)

These YAML files describe reproducible experiments. Always inspect the resolved
configuration before training.

## Learning Minimal

`learning_minimal.yaml` is the first run: CPU-friendly sample limits, 128 px
inputs, a narrow U-Net, two epochs, and cross-entropy. It proves the workflow;
its score is not the reference result.

```bash
uv run segment show-config --config configs/learning_minimal.yaml
uv run segment train --config configs/learning_minimal.yaml --dry-run
```

## Reference U-Net

`reference_unet.yaml` is the fixed 25-epoch measured run. It uses the complete
prepared train/validation splits and a CE plus foreground Dice objective. Do not
edit it in place when comparing an idea; copy it or use a named override.

## DeepLabV3 Comparison

`deeplabv3_resnet50.yaml` is an advanced architecture comparison. Its
pretrained setting may download torchvision weights. Record whether weights
were used; a pretrained model is not a clean architecture-only comparison with
randomly initialized U-Net.

## Safe Overrides

Use `--set` for one deliberate variable and assign a distinct `run_name`.
Image size must be divisible by 16. Model classes, raw labels, ignore index, and
preprocessing become checkpoint compatibility data.

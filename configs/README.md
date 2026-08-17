# Three Runnable Configurations

[简体中文](README.zh-CN.md) | [Configuration reference](../docs/reference/config-reference.md)

Start from the nearest working config, inspect it with `show-config`, and change
one named setting instead of writing YAML from scratch.

| Config | Purpose | Main settings |
| --- | --- | --- |
| `learning_minimal.yaml` | first workflow | 128 px, bounded samples, 2 epochs, CE, narrow U-Net |
| `reference_unet.yaml` | full handwritten run | 256 px, 25 epochs, base 32, CE+Dice, cosine |
| `deeplabv3_resnet50.yaml` | advanced comparison | pretrained DeepLabV3, batch 4, 15 epochs |

```bash
uv run segment show-config --config configs/learning_minimal.yaml
uv run segment train --config configs/learning_minimal.yaml --dry-run
```

Do not edit the reference config in place for a comparison. Use `--set` and a
distinct run name. Default DeepLab versus U-Net changes initialization, model,
batch, and LR, so describe it as a complete-system comparison rather than a pure
architecture ablation.

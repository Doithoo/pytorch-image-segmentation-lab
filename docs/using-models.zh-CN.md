# 切换模型

[English](using-models.md) | [可用模型](model-zoo.zh-CN.md)

先看最终配置，再做干跑：

```bash
uv run segment show-config --config configs/fcn_resnet50.yaml
uv run segment train --config configs/fcn_resnet50.yaml --dry-run --device cpu
```

临时切换注册模型时，要同时清掉原模型专用参数：

```bash
uv run segment train --config configs/learning_minimal.yaml --dry-run \
  --set model.name lraspp_mobilenet_v3_large \
  --set model.params '{}'
```

`expected_num_classes` 要与准备目录中的 `dataset.yaml` 一致。第一次用新模型建议设置
`pretrained: false`，避免命令中途下载权重。显存不足时依次减小 batch、输入尺寸、模型
宽度或 encoder 规模。每次比较使用不同 `run_name`；只有标签模式哈希相同时，才直接比较
`segment compare-runs ... --metric valid_mean_iou` 的数值。

可直接参考 `configs/fcn_resnet50.yaml`、`configs/lraspp_mobilenet.yaml` 和
`configs/smp_unetplusplus.yaml`。

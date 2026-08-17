# Switching models

[中文](using-models.zh-CN.md) | [Model zoo](model-zoo.md)

Inspect the resolved configuration before training:

```bash
uv run segment show-config --config configs/fcn_resnet50.yaml
uv run segment train --config configs/fcn_resnet50.yaml --dry-run --device cpu
```

For a quick switch, override the registered name and clear model-specific parameters:

```bash
uv run segment train --config configs/learning_minimal.yaml --dry-run \
  --set model.name lraspp_mobilenet_v3_large \
  --set model.params '{}'
```

Check `expected_num_classes` against prepared `dataset.yaml`. Start with `pretrained: false`
to avoid an unexpected download. Reduce batch size first after an OOM, then image size,
then model width or encoder size. Use a distinct `run_name` for every comparison and use
`segment compare-runs ... --metric valid_mean_iou` only when label schema hashes match.

The maintained examples are `configs/fcn_resnet50.yaml`, `configs/lraspp_mobilenet.yaml`,
and `configs/smp_unetplusplus.yaml`.

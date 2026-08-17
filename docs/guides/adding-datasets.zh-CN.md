# 接入自己的 Dataset

[English](adding-datasets.md) | [示例源码](../../examples/extensions/my_dataset.py) | [文档导航](../README.zh-CN.md)

特殊文件格式无法用内置 provider 表达时，设置 `data.factory`：

```yaml
data:
  name: custom_synthetic
  factory: examples.extensions.my_dataset:create_dataset
  params: {length: 8, source_size: 48}
```

工厂接收 `split`、`data_dir`、`manifest_dir`、`transform`、`label_schema` 和自定义参数，
为每个 split 返回非空 `Dataset`。样本必须是 `(float_image, long_target, sample_id)`，形状
分别为 `[C,H,W]` 和 `[H,W]`。读出原图和掩码后调用传入的成对 `transform`，不要在三套
split 中各自复制缩放与归一化逻辑。

外部 Dataset 仍以准备目录的 `dataset.yaml` 作为标签和切分合同。训练前先准备或生成该
元数据，然后用 `--dry-run` 检查一个真实 batch。

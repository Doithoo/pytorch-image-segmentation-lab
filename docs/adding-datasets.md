# Adding a Dataset factory

[中文](adding-datasets.zh-CN.md) | [Example source](../examples/extensions/my_dataset.py)

Set `data.factory` when files cannot be represented by a maintained provider:

```yaml
data:
  name: custom_synthetic
  factory: examples.extensions.my_dataset:create_dataset
  params: {length: 8, source_size: 48}
```

The factory receives `split`, `data_dir`, `manifest_dir`, `transform`, `label_schema`, and params.
It returns a non-empty `Dataset` for each split. Every item is `(float_image, long_target,
sample_id)` with shapes `[C,H,W]` and `[H,W]`. Call the provided paired `transform` after reading
the source image and mask so training and inference geometry remains consistent.

External factories still use prepared `dataset.yaml` as the label and split contract. Prepare or
write that metadata before training, then run the config with `--dry-run`.

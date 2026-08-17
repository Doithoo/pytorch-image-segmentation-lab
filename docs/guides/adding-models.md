# Adding your model

[中文](adding-models.zh-CN.md) | [Example source](../../examples/extensions/my_segmenter.py) | [Documentation](../README.md)

An external factory receives keyword arguments and returns `torch.nn.Module`:

```python
def create_model(*, in_channels, num_classes, pretrained, **params): ...
```

Its forward method must return logits shaped `[B, num_classes, H, W]`. Do not apply softmax;
the loss consumes logits. Configure the runnable example as follows:

```yaml
model:
  name: custom_direct_logits
  factory: examples.extensions.my_segmenter:create_model
  pretrained: false
  params: {width: 12}
```

```bash
uv run segment train --config configs/custom_model_example.yaml --dry-run --device cpu
```

The loader checks the factory signature, return type, output type, channels, and spatial size.
Keep the module importable from the repository root or install your own package. Save architecture
choices in `params`; this keeps checkpoints and `run.yaml` understandable.

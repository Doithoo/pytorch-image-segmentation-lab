# 接入自己的模型

[English](adding-models.md) | [示例源码](../examples/extensions/my_segmenter.py)

外部工厂接收关键字参数并返回 `torch.nn.Module`：

```python
def create_model(*, in_channels, num_classes, pretrained, **params): ...
```

`forward` 必须直接返回 `[B, num_classes, H, W]` logits，不要先做 softmax。配置可照着
可运行示例写：

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

加载器会检查工厂签名、返回类型、logits 通道和空间尺寸。模块需要能从仓库根目录导入，
也可以先安装成自己的包。把结构选择写在 `params` 中，checkpoint 和 `run.yaml` 才能说明
这次实际创建了什么模型。

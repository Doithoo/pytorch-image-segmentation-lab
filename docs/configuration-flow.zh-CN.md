# 配置流

[English](configuration-flow.md) | [返回 README](../README.zh-CN.md)

配置显式且严格，在加载数据或构建模型前只解析一次。

## 优先级

```text
code defaults < YAML file < --set override < dedicated CLI option
```

后面的层级覆盖前面的值。`--device` 等专用参数适合一次性运行选择；YAML 应描述可
复现实验；`--set` 适合进行一个受控变化。

## 运行前检查

```bash
uv run segment show-config --config configs/learning_minimal.yaml
uv run segment show-config --config configs/learning_minimal.yaml \
  --set train.epochs 3 --set model.base_channels 8
```

第一条命令打印所有最终字段。第二条展示带类型的点路径覆盖。未知路径、无效布尔值、
不能整除的图像尺寸和不兼容标签映射都会立即失败。

## 命令专用覆盖

```bash
uv run segment train --config configs/learning_minimal.yaml \
  --set train.epochs 3 --device cpu --output-dir artifacts
```

这里 `--device cpu` 和 `--output-dir artifacts` 优先级最高。保存的 `config.yaml`
是最终解析结果，而不只是输入 YAML 的副本。

## Checkpoint 恢复

恢复训练会先校验模型构造、原始标签映射、忽略索引以及训练/验证清单哈希，再加载
优化器状态。评估和推理从 checkpoint 元数据重建模型与预处理，不依赖当前默认值。

字段含义参见[配置参考](config-reference.zh-CN.md)，安全对照方法参见
[实验指南](experiments.zh-CN.md)。

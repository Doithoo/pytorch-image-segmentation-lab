# 使用自己的数据

[English](using-your-data.md) | [返回 README](../README.zh-CN.md)

自定义数据与 Oxford 使用同一个清单驱动 Dataset。准备命令只要求文件名成对，并且
掩码语义明确。

## 安排配对文件

```text
my-data/
|-- images/
|   |-- sample-001.jpg
|   `-- sample-002.png
`-- masks/
    |-- sample-001.png
    `-- sample-002.png
```

图像和掩码的文件名主干必须完全相同。只有相对主干仍匹配时才允许嵌套路径。掩码
必须是单通道整数图，不能使用彩色截图或透明叠加图。

## 定义标签语义

修改配置前先检查唯一像素值。`data.label_map` 把原始值映射为连续类别索引，
`data.ignore_values` 列出既不参与损失也不参与指标的像素。每个原始值都必须有明确
去向，否则加载会失败，而不是暗中当作背景。

首个版本只支持二类：模型类别 0 是背景，类别 1 是前景。多类、实例 ID、多边形和
软掩码需要修改代码，默认流水线不会接收。

## 生成清单

```bash
uv run segment prepare-data --config configs/learning_minimal.yaml \
  --data-dir my-data --manifest-dir my-manifests \
  --source-format paired
uv run python scripts/preview_dataset.py my-manifests/train.csv \
  --output artifacts/my-data-preview.png --limit 4
```

通用配对数据根据配置中的比例和种子确定性划分训练/验证/测试集。准备过程会校验
缺失配对、重复内容、不可读文件、图像/掩码尺寸不一致和意外标签值。训练前阅读
`source.yaml` 和 `summary.txt`。

## 调整预览

内置预览使用 Oxford 的 `1/2/3` 原始调色规则。若编码不同，先使用生成的清单和
Dataset 测试，再调整 `scripts/preview_dataset.py` 中的小型映射常量。绝不能把
预览渲染结果当作训练掩码。

## 安全训练

复制 `learning_minimal.yaml`，设置 `data.manifest_dir`、标签映射和忽略值，然后
运行 `show-config` 和 `--dry-run`。从小子集开始，在增加轮数前确认图像与目标形状
对齐，且映射值只有 `0`、`1` 和忽略索引。

保留最终测试集，不要用于调参。如果数据含患者、视频或重复主体等分组，应编写按组
划分器；随机按图划分可能泄漏近似内容并夸大结果。

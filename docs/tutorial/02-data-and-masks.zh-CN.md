# 教程 02：数据与掩码

[English](02-data-and-masks.md) | [教程索引](../learning-path.zh-CN.md)

分割质量始于对齐且解释正确的标签。

## 下载与校验

```bash
uv run python scripts/download_data.py --data-dir data/raw
uv run python scripts/download_data.py --data-dir data/raw --verify-only
```

下载先写入 `.part` 文件，只有完整且 SHA-256 匹配的归档才替换缓存。解压会拒绝
逃逸到目标目录之外的路径。

## 保持官方测试集

```bash
uv run segment prepare-data --config configs/learning_minimal.yaml \
  --data-dir data/raw --manifest-dir data/manifests \
  --source-format oxford-pet
```

官方测试列表原样保留，上游 trainval 样本在每个品种内按 80/20 划分。阅读
`source.yaml`、`summary.txt` 和 CSV 行；其哈希会把后续 checkpoint 绑定到准确成员。

## 检查原始与映射掩码

```bash
uv run python scripts/preview_dataset.py data/manifests/train.csv \
  --output artifacts/dataset_preview.png --limit 4
uv run python examples/02_paired_transform.py
```

确认原始值 1、2、3 变为宠物 1、背景 0、忽略 255。确认翻转和缩放使图像边缘与
掩码边缘保持对齐。

## 发现泄漏与损坏

跨划分的完全重复内容会报错。缺失配对、不可读文件、尺寸不匹配和未知掩码值也会
尽早失败。这些检查无法发现语义相关主体；自定义医疗或视频数据需要按组划分。

下一章：[教程 03](03-unet.zh-CN.md)。

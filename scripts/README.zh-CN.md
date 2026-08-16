# 实用脚本

这些脚本把数据集准备和实验产物生成过程清晰地展示给初学者。请激活项目环境，
并从仓库根目录运行脚本。

## 下载 Oxford-IIIT Pet

```bash
uv run python scripts/download_data.py --data-dir data/raw
```

归档会先以临时文件形式写入 `data/downloads`。安装器仅在两个 SHA-256 校验均
通过后，才创建 `data/raw/images`、`data/raw/masks` 和
`data/raw/annotations`。使用 `--verify-only` 可以离线校验缓存且不执行安装；
只有明确需要重新下载时才使用 `--force`。

## 预览清单

```bash
uv run python scripts/preview_dataset.py data/manifests/train.csv \
  --output artifacts/dataset_preview.png --limit 4
```

每行依次展示原图、Oxford 原始 trimap、映射后的训练掩码和前景叠加图。脚本只
读取源文件，不会修改数据。

## 绘制训练指标

```bash
uv run python scripts/plot_metrics.py artifacts/my-run/metrics.csv \
  --output artifacts/my-run/loss_curve.png
```

曲线展示每个 epoch 记录的训练损失和验证损失。

英文版参见 [Utility Scripts](README.md)。

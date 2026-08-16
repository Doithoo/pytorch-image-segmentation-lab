# 数据和结果辅助脚本

[English](README.md) | [返回 README](../README.zh-CN.md)

三个脚本分别处理下载、训练前预览和训练后曲线。都从仓库根目录通过 `uv run` 执行。

## 下载并核对 Oxford

```bash
uv run python scripts/download_data.py --data-dir data/raw
uv run python scripts/download_data.py --data-dir data/raw --verify-only
```

下载先写 `.part`，SHA-256 正确后才替换缓存；两个归档都验证通过后才安装。`--force`
会重新下载，只在确认缓存需要替换时使用。

## 训练前看一眼数据

```bash
uv run python scripts/preview_dataset.py data/manifests/train.csv \
  --output artifacts/dataset_preview.png --limit 4
```

每行依次显示原图、Oxford trimap、映射后的训练 mask 和叠加图。打开文件检查边缘对齐，
不要把“脚本执行成功”当成“标签一定正确”。自定义编码需要调整预览显示，不能修改原始
训练 mask 去迎合颜色。

## 训练后画 loss

```bash
uv run python scripts/plot_metrics.py artifacts/first-unet/metrics.csv \
  --output artifacts/first-unet/loss_curve.png
```

曲线只读取 `metrics.csv`，显示 train/valid loss。结合 validation pet IoU 和叠加图解释，
不要只凭两条 loss 曲线选择最终模型。

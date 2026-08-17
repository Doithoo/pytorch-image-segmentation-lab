# 数据集格式

[English](dataset-formats.md) | [使用自己的数据](using-your-data.zh-CN.md)

准备数据前先运行 `uv run segment list-datasets` 和 `dataset-info 数据源名`。

| provider | 原始数据形式 |
| --- | --- |
| `oxford_pet` | Oxford 官方图片、trimap 和切分文件 |
| `paired_masks` | 相对路径同名的 `images/`、`masks/` 目录树 |
| `split_folders` | 已经分好的 `train/valid/test` 目录 |
| `manifest` | 三个 split 的现成 CSV |
| `voc2012` | 解压后的 VOC 2012 目录结构 |

准备结果统一包含 `dataset.yaml` 和 train/valid/test manifest。元数据记录标签模式、路径、
样本数和哈希。内置 provider 接受 `L/P/I;16` 索引掩码；RGB 颜色掩码目前需要通过外部
Dataset factory 转为类别 ID。几何处理支持 `stretch`、`letterbox` 和成对 crop；掩码
始终使用最近邻插值。

需要由 Python 代码读取特殊格式时，继续看[接入自己的 Dataset](adding-datasets.zh-CN.md)。

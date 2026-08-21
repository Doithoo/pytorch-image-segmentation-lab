# 教程 02：训练之前，先相信你的掩码

[English](02-data-and-masks.md) | [教程索引](README.zh-CN.md)

分割模型会非常认真地学习错误标签。长训练之前花十分钟看图片、掩码和划分，通常比
换模型更有价值。

## 下载时发生了什么

```bash
uv run python scripts/download_data.py --data-dir data/raw
```

下载器会把 Oxford 图片和标注归档写入 `data/downloads`，核对固定 SHA-256，再安装到
`data/raw`。未完成的内容先写入 `.part`，不会覆盖完整缓存；解压会拒绝目录穿越、
符号链接和硬链接。

已有缓存时可以离线复查：

```bash
uv run python scripts/download_data.py --data-dir data/raw --verify-only
```

数据文件沿用 Oxford-IIIT Pet 的上游许可，不提交到仓库。

## 为什么不让训练代码直接扫描目录

先生成固定 manifest：

```bash
uv run segment prepare-data --config configs/learning_minimal.yaml \
  --data-dir data/raw --manifest-dir data/manifests \
  --source-format oxford-pet
```

`train.csv`、`valid.csv` 和 `test.csv` 明确记录每个样本属于哪个 split。这样模型、loss
或增强方式变化时，数据成员仍然相同，实验才可以比较。

Oxford 官方 test 列表保持不变；trainval 在每个品种内按 80/20 切成 train/valid。
`source.yaml` 记录来源、标签语义、样本数和 manifest SHA-256；`summary.txt` 提供适合
终端快速阅读的摘要。

```bash
sed -n '1,20p' data/manifests/summary.txt
sed -n '1,5p' data/manifests/train.csv
```

CSV 使用相对路径，因此数据目录与 manifest 一起移动后仍可解析。

## 看懂原始 trimap

Oxford 原始值是：

```text
1 = pet
2 = background
3 = unclassified border
```

训练前映射为：

```text
1 = foreground class
0 = background class
255 = ignore index
```

边缘值 3 不是“第三个边界类别”。它表示这部分不参与判断：交叉熵跳过 255，Dice 和
指标也过滤相同像素，模型仍然只输出背景/宠物两个通道。

## 用预览图发现问题

```bash
uv run python scripts/preview_dataset.py data/manifests/train.csv \
  --output artifacts/dataset_preview.png --limit 4
```

![图片、原始 trimap、映射掩码和叠加图](../assets/reference-unet-dataset-preview.png)

逐行检查四列：

1. Image：图片是否能打开，主体是否符合任务。
2. Raw mask：白色边界与宠物轮廓是否合理。
3. Mapped mask：绿色是宠物，黑色是背景，灰色是忽略区域。
4. Overlay：边缘是否与原图主体对齐。

预览不能证明所有数据正确，但能快速发现通道、颜色、错配和标签语义错误。

## 成对变换为什么必须共用随机选择

```bash
uv run python examples/02_paired_transform.py
```

示例左半部分为红色/宠物，强制水平翻转后，两者应同时移动到右侧。真实训练中，随机翻转
只能采样一次，再同时应用到 image 和 mask。若分别采样，约一半样本会错位。

缩放也要区别处理：

| 对象 | 插值 | 原因 |
| --- | --- | --- |
| RGB image | bilinear + antialias | 连续颜色适合平滑 |
| class mask | nearest-neighbor | 只能复制已有类别 ID |

默认 `stretch` 会把不同长宽比统一拉伸到固定形状，便于初学者追踪张量，但会改变物体
比例。长宽比敏感的数据可改用 `letterbox`，它保持比例并用 ignore index 填充掩码；训练
还可设置 `crop_size` 做成对随机裁剪。无论哪种模式，图片与掩码必须共享同一组几何参数。

## 准备程序会检查什么

- 缺失的图片/掩码配对；
- 图片或掩码无法读取；
- 两者尺寸不一致；
- 掩码不是单通道；
- 出现配置未覆盖的标签值；
- 图片和掩码内容完全重复；
- 通用数据比例产生空的 train、valid 或 test split。

它无法自动发现“同一患者出现在两个 split”“相邻视频帧泄漏”或语义相近的重复主体。
这类数据必须先按患者、视频或来源分组，再生成 manifest。

## 一个值得亲手做的检查

任选一张原始掩码，打印唯一值：

```bash
uv run python -c "from pathlib import Path; import numpy as np; from PIL import Image; p=next(Path('data/raw/masks').glob('*.png')); print(p, np.unique(np.asarray(Image.open(p))))"
```

然后在 `src/image_segmenter/data/dataset.py` 找到 `map_mask`，回答：若出现值 4，代码是
把它当背景、忽略，还是立即失败？为什么立即失败对自定义数据更安全？

准备读模型时进入[教程 03：U-Net](03-unet.zh-CN.md)。换自己的数据请看
[自己的数据指南](../guides/using-your-data.zh-CN.md)。

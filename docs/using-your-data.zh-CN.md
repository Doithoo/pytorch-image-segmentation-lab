# 换成自己的图片和掩码

[English](using-your-data.md) | [返回 README](../README.zh-CN.md)

通用入口从同名图片/掩码对开始，也支持从 0 开始的连续多分类标签。先用很小的数据副本
完成预览和 dry-run，再指向完整数据；不要第一次就启动长训练。

## 整理目录

```text
my-data/
├── images/
│   ├── sample-001.jpg
│   └── group-a/sample-002.png
└── masks/
    ├── sample-001.png
    └── group-a/sample-002.png
```

图片与掩码的相对文件名主干必须相同。`images/group-a/a.jpg` 对应
`masks/group-a/a.png`；缺失、重复或尺寸不同都会在准备阶段报错。

掩码必须是单通道整数图，不是彩色截图、透明叠加图或概率热力图。

## 先确认原始像素值

任选几张 mask 打印唯一值：

```bash
uv run python -c "from pathlib import Path; import numpy as np; from PIL import Image; [print(p, np.unique(np.asarray(Image.open(p)))) for p in list(Path('my-data/masks').rglob('*.png'))[:5]]"
```

假设你的数据使用 `0=background`、`255=object`，没有忽略值，配置应写：

```yaml
data:
  label_map: {0: 0, 255: 1}
  ignore_values: []
loss:
  ignore_index: 254
```

`ignore_index` 只需与训练类别 0/1 不同；即使数据没有忽略像素，也要保留一个不会与
类别冲突的值。每个可能出现的原始值必须进入 `label_map` 或 `ignore_values`，未知值会
立即失败，而不是自动变成背景。

## 为新数据复制一份配置

不要修改参考配置原件。创建例如 `configs/my_data.yaml`，至少调整：

```yaml
data:
  data_dir: my-data
  manifest_dir: data/my-manifests
  image_size: 128
  label_map: {0: 0, 255: 1}
  ignore_values: []
  split_ratios: [0.8, 0.1, 0.1]
  num_workers: 0
  max_train_samples: 64
  max_valid_samples: 16
  max_test_samples: 16
loss:
  ignore_index: 254
run_name: my-data-check
```

先用低分辨率和样本上限验证流程。确认掩码与指标正确后，再移除上限或增加模型宽度。

## 生成独立 manifest

```bash
uv run segment prepare-data --config configs/my_data.yaml \
  --data-dir my-data --manifest-dir data/my-manifests \
  --source-format paired
```

通用入口按配置 seed 和比例确定性打乱后切分。如果样本数量与比例会产生空的 train、
valid 或 test，命令会要求增加样本或调整比例。完整学习流程需要三个非空 split。

准备完成后检查：

```bash
sed -n '1,20p' data/my-manifests/summary.txt
sed -n '1,5p' data/my-manifests/train.csv
```

不要覆盖 Oxford 的 `data/manifests`，否则参考实验与新数据实验的成员身份会混在一起。

## 预览与 dry run

内置 preview 的原始颜色面板按 Oxford `1/2/3` 设计，不适合所有编码。对自定义数据，
至少用 Dataset 读取样本并检查映射后的唯一值；需要长期使用时，再让预览脚本读取配置
中的 label map，而不是把训练掩码改成 Oxford 颜色。

```bash
uv run segment train --config configs/my_data.yaml --dry-run --device cpu
```

确认输出：图片 `[B,3,H,W]`、mask `[B,H,W]`、logits `[B,2,H,W]`；mask values 只包含
0、1 和你的 ignore index；valid pixel ratio 合理。

## 当前 resize 策略的限制

图片和掩码会一起拉伸为 `image_size x image_size`。普通照片常可作为简单基线，但长宽比
敏感的数据可能失真：细胞形态、道路宽度、文档字符或工业尺寸都可能受影响。

若形变不可接受，应在 `PairedTransform` 中实现并测试 padding/cropping，同时保证 image
与 mask 使用相同几何参数，mask 仍用最近邻插值。记录最终预处理到 checkpoint。

## 避免数据泄漏

随机按图片切分不适合以下情况：

- 同一个患者的多张扫描；
- 同一段视频的相邻帧；
- 同一物体的连拍或不同裁剪；
- 同一地点、批次、设备生成的高度相关样本。

这些数据应先按主体/来源分组，再把组分配到 split。否则 validation/test 看起来很好，
只是因为模型见过近似内容。

## 跑一个短基线

```bash
uv run segment train --config configs/my_data.yaml \
  --set train.epochs 3 --set run_name my-data-baseline
uv run segment evaluate --checkpoint artifacts/my-data-baseline/best.pt \
  --split valid --plot --device cpu
```

先看错误图和前景比例，再决定下一步。不要遇到低 IoU 就立即换最大模型；标签语义、
划分、形变和目标大小通常更值得先检查。

## 常见问题

| 现象 | 先检查 |
| --- | --- |
| missing pair | 图片和掩码相对主干是否完全相同 |
| unexpected values | 原始唯一值是否全部写进 label/ignore 配置 |
| empty split | 样本量与比例是否能让三个 split 都非空 |
| valid 很好、test 很差 | 是否按相关主体分组，是否分布变化 |
| accuracy 高、pet IoU 低 | 前景是否很小，模型是否只预测背景 |
| overlay 整体错位 | 原图/掩码尺寸和几何变换是否一致 |

更多错误见[排错指南](troubleshooting.zh-CN.md)，实验记录方式见
[实验指南](experiments.zh-CN.md)。

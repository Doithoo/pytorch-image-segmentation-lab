# PyTorch 图像分割学习实验室

[English](README.md)

这是一个小而完整的语义分割学习项目，适合已经接触过 PyTorch 张量和图像分类、
准备理解像素级预测的学习者。主线使用手写 U-Net 和 Oxford-IIIT Pet，并完整展示
成对变换、原始标签映射、忽略像素、损失、前景指标、checkpoint、评估与推理产物。

本仓库优先保证学习价值和可复现性，而不是追逐榜单成绩。首个版本只支持二类语义
分割，实例分割和全景分割有意不在范围内。

## 你将构建什么

一张 RGB 图像会变成每个像素的两个 logits：背景和宠物。训练采用 Oxford 官方
trimap，并固定使用以下映射：

```text
1 -> pet
2 -> background
3 -> ignored/unclassified (training index 255)
```

原始值 3 表示数据集中的未分类区域，不是第三个语义“边界类别”，既不参与损失，
也不参与指标。最佳 checkpoint 由验证集宠物 IoU 选择；未改动的官方测试集只用于
最终报告。

## 快速开始

需要 Python 3.10-3.12 和 [uv](https://docs.astral.sh/uv/)。默认学习配置可在 CPU
运行，且不会下载预训练权重。

```bash
git clone https://github.com/Doithoo/pytorch-image-segmentation-lab.git
cd pytorch-image-segmentation-lab
uv sync --extra dev
```

下载并校验 Oxford-IIIT Pet，然后生成确定性清单。官方测试成员保持不变；上游
trainval 样本按品种分别以 80/20 划分为训练集和验证集。

```bash
uv run python scripts/download_data.py --data-dir data/raw
uv run segment prepare-data --config configs/learning_minimal.yaml \
  --data-dir data/raw --manifest-dir data/manifests \
  --source-format oxford-pet
uv run python scripts/preview_dataset.py data/manifests/train.csv \
  --output artifacts/dataset_preview.png --limit 4
```

正式训练前先验证一个完整优化批次，再运行两轮小型学习实验。

```bash
uv run segment train --config configs/learning_minimal.yaml --dry-run
uv run segment train --config configs/learning_minimal.yaml
```

评估保存的 checkpoint，并对一张图像预测掩码。请将示例路径替换为训练输出的
实际路径。

```bash
uv run segment evaluate --checkpoint artifacts/learning-unet/best.pt \
  --split test --plot --device cpu
uv run segment predict --checkpoint artifacts/learning-unet/best.pt \
  --image data/raw/images/Abyssinian_1.jpg \
  --output artifacts/predictions/Abyssinian_1.png --device cpu
```

## 分步学习

先运行 [examples](examples/README.zh-CN.md) 中的五个文件，再沿着
[学习路径](docs/learning-path.zh-CN.md) 从张量形状走到完整实验。六章
[教程](docs/tutorial/00-basics.zh-CN.md) 用概念、检查点和思考题讲解同一路径。

常用地图：

- [系统如何工作](docs/how-it-works.zh-CN.md)
- [代码导览](docs/code-tour.zh-CN.md)
- [配置流](docs/configuration-flow.zh-CN.md)
- [完整配置参考](docs/config-reference.zh-CN.md)
- [使用自己的成对数据](docs/using-your-data.zh-CN.md)
- [进行受控实验](docs/experiments.zh-CN.md)
- [实测参考实验](docs/reference-run.zh-CN.md)

## 可以检查的产物

训练会在单个运行目录中写入解析后的配置、环境元数据、每轮指标以及原子保存的
`best.pt`/`last.pt`。评估会增加全局和逐图表格、示例对比、最差案例及可选损失
曲线。推理会把机器可读的类别索引掩码与彩色掩码、叠加图分开保存。

主指标是宠物前景 IoU。背景占比高时，像素准确率可能很好，但宠物掩码仍很差，
因此它绝不用于选择 checkpoint。如果单张图的目标和预测都没有前景，IoU/Dice
没有定义；项目会计数并从逐图均值中排除，而不是暗中当作满分。

## 配置

- `configs/learning_minimal.yaml`：限制样本的 CPU 入门配置，128 像素、两轮、
  仅交叉熵。
- `configs/reference_unet.yaml`：已实测的 25 轮手写 U-Net 配置。
- `configs/deeplabv3_resnet50.yaml`：进阶 torchvision 对照；预训练权重可能需要
  单独联网下载。

配置优先级为 `默认值 < YAML < --set < 专用 CLI 参数`。未知键和无效组合会在
耗时操作之前失败。用以下命令检查最终值：

```bash
uv run segment show-config --config configs/learning_minimal.yaml
uv run segment --help
```

## 可复现性与限制

清单记录划分来源和 SHA-256。Checkpoint 保存最终配置、标签映射、预处理、清单
校验和、优化器状态和随机状态。随机种子可以减少差异，但设备内核和 PyTorch 版本
仍可能改变浮点结果；应比较趋势和报告条件，而不是只盯一个小数分数。

参考结果只是当前实现和记录环境的证据，不保证每台机器都完全相同。Oxford-IIIT
Pet 图像和标注沿用上游许可与署名；数据只下载到本地，不提交到本仓库。

## 开发

```bash
uv run ruff check .
uv run pytest
uv build
uv run twine check dist/*
```

提交修改前请阅读 [CONTRIBUTING.md](CONTRIBUTING.zh-CN.md)。测试只使用本地合成
fixture，不会下载数据集或预训练模型。

项目采用 [MIT License](LICENSE)。

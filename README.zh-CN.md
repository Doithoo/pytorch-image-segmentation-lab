# PyTorch 图像分割学习实验室

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![CI](https://github.com/Doithoo/pytorch-image-segmentation-lab/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)

**English version: [README.md](README.md)**

通过一个可以真实运行的 PyTorch 项目，学习怎样把一张图片变成逐像素的宠物掩码。
主线使用 Oxford-IIIT Pet、手写 U-Net 和二类语义分割，覆盖数据检查、成对变换、
训练、IoU/Dice 评测、错误可视化和单图推理。

![Oxford-IIIT Pet 图像、原始 trimap、训练掩码和叠加图](docs/assets/reference-unet-dataset-preview.png)

## 这个项目适合谁

适合已经接触过 Python、PyTorch Tensor、loss 和梯度，但还没有完整做过图像分割项目的
学习者。不要求会手写 U-Net，也不要求先读论文。

推荐学习时间约 **8–12 小时**。第一次先跑通一条短流程，再回头理解标签映射、skip
connection、Dice loss 和全局/逐图指标。若 logits 和交叉熵仍然陌生，从
[分割所需的最小概念](docs/tutorial/00-basics.zh-CN.md)开始。

第一条路线仍是二类语义分割：类别 0 是背景，类别 1 是宠物。跑通之后，同一套管线可
使用多分类标签模式、五种数据 provider、九个注册模型，也可通过 Python 工厂接入自己的
模型或 Dataset。实例分割和全景分割不在当前范围内。

## 从这里开始

第一次只完成下面这条路径：

```text
下载数据 -> 生成固定清单 -> 看一眼掩码 -> dry run -> 训练 -> 评测 -> 推理
```

需要 Python 3.10–3.12 和 [uv](https://docs.astral.sh/uv/)。安装锁定的开发环境：

```bash
git clone https://github.com/Doithoo/pytorch-image-segmentation-lab.git
cd pytorch-image-segmentation-lab
uv sync --extra dev
uv run segment --version
```

下载并校验 Oxford-IIIT Pet，然后按官方列表保留测试集；上游 trainval 样本会按品种
分别切成训练集和验证集。

```bash
uv run python scripts/download_data.py --data-dir data/raw
uv run segment prepare-data --config configs/learning_minimal.yaml \
  --data-dir data/raw --manifest-dir data/manifests \
  --source-format oxford-pet
uv run python scripts/preview_dataset.py data/manifests/train.csv \
  --output artifacts/dataset_preview.png --limit 4
```

打开 `artifacts/dataset_preview.png`，先确认图片边缘和掩码边缘对齐。原始 trimap 的
三个值会变成：

```text
1 -> pet
2 -> background
3 -> ignored/unclassified (training index 255)
```

值 3 是不参与训练和指标的未分类区域，不是模型需要预测的第三类。

## 先检查一个 batch

正式训练前执行一次真实的前向、损失、反向和参数更新：

```bash
uv run segment train --config configs/learning_minimal.yaml --dry-run
```

日志应包含类似内容：

```text
images: (8, 3, 128, 128)
masks: (8, 128, 128) values=[0, 1, 255]
logits: (8, 2, 128, 128)
loss: ...
dry-run OK
```

`dry-run OK` 只说明数据、模型、损失和反向传播已经连通，不表示模型学会了分割。

接着运行两轮小实验：

```bash
uv run segment train --config configs/learning_minimal.yaml \
  --set run_name first-unet
```

先打开 `artifacts/first-unet/config.yaml` 和 `metrics.csv`。前者回答“这次到底用了什么
配置”，后者记录训练损失和验证集宠物 IoU 怎样变化。

## 看看模型画出了什么

在留出的测试集评测最佳 checkpoint，并预测一张图片：

```bash
uv run segment evaluate --checkpoint artifacts/first-unet/best.pt \
  --split test --plot --device cpu
uv run segment predict --checkpoint artifacts/first-unet/best.pt \
  --image data/raw/images/Abyssinian_1.jpg \
  --output artifacts/predictions/Abyssinian_1.png --device cpu
```

评测不只给一个分数，还会保存逐类表格、逐图表格、代表样本和最差案例。推理分别保存
机器可读的 0/1 掩码、彩色掩码和叠加图，避免把展示图片误当成训练标签。

![U-Net 预测、目标和像素错误](docs/assets/reference-unet-overlay.png)

主指标是验证集和测试集的宠物前景 IoU。背景很多时，pixel accuracy 可能很好看，
但宠物轮廓仍然很差；因此项目使用 validation pet IoU 选择 `best.pt`，并让 official
test split 留到最终报告。

## 一次真实参考运行

仓库记录了一次 25 epoch 的 U-Net 实验。最佳 checkpoint 来自第 24 轮，验证集宠物
IoU 为 `0.902836`；在 3,669 张官方测试图片上，宠物 IoU 为 `0.906669`，Dice 为
`0.951050`。这些数字只适用于记录的数据清单、配置、代码 revision 和运行环境。

![参考运行的训练与验证损失](docs/assets/reference-unet-loss-curve.png)

[参考运行说明](docs/recorded-run/README.zh-CN.md)保留配置、环境、清单哈希、逐类/逐图指标和
失败案例。当前仓库树不再保留 89 MB 的模型 checkpoint；重新训练时比较配置、manifest
哈希和结果趋势，另行获得原 checkpoint 时再用记录的 SHA-256 校验文件。

## 接着学什么

- 从[文档导航](docs/README.zh-CN.md)按用途选择操作指南、查询资料、原理说明或实测记录。
- 按[学习路线](docs/tutorial/learning-path.zh-CN.md)逐步实践，或在[教程索引](docs/tutorial/README.zh-CN.md)中按问题选择章节。
- 用[代码导览](docs/concepts/code-tour.zh-CN.md)追踪一张图片，再用[原理说明](docs/concepts/how-it-works.zh-CN.md)理解背后的设计。
- 命令、数据、训练或推理失败时直接查[排错指南](docs/guides/troubleshooting.zh-CN.md)。

## 换成自己的数据

自定义数据使用同名图片/掩码对：

```text
my-data/
├── images/sample-001.jpg
└── masks/sample-001.png
```

先阅读[自己的数据指南](docs/guides/using-your-data.zh-CN.md)。它说明怎样检查掩码唯一值、
设置 `label_map`、生成独立 manifest、避免空切分，并在长训练前验证一个 batch。
内置成对几何处理支持 stretch、letterbox、crop 和 flip；长宽比敏感的数据先比较
stretch 与 letterbox，再确定后续配置。

## 配置示例

- `configs/learning_minimal.yaml`：CPU 可运行的 128 像素、两轮、小样本入门配置。
- `configs/reference_unet.yaml`：已经实测的 25 轮手写 U-Net 配置。
- `configs/deeplabv3_resnet50.yaml`：进阶 torchvision 对照，预训练权重需要联网下载。
- `configs/fcn_resnet50.yaml`、`configs/lraspp_mobilenet.yaml`：不下载权重的 torchvision 起点。
- `configs/smp_unetplusplus.yaml`：使用可选 SMP 依赖，encoder 权重设为 null。
- `configs/custom_model_example.yaml`、`configs/custom_dataset_example.yaml`：可导入的工厂示例。

配置优先级是：

```text
代码默认值 < YAML 文件 < --set 覆盖 < 专用命令行参数
```

```bash
uv run segment show-config --config configs/learning_minimal.yaml
uv run segment --help
```

## 项目结构

```text
examples/                    先运行和阅读的 5 个小程序
configs/                     入门、参考和模型对照配置
docs/                        学习路线、教程、原理和实验指南
scripts/                     下载、预览和绘图工具
src/image_segmenter/         安装后运行的应用和可复用代码
tests/                       单元、集成、CLI 和端到端测试
data/manifests/              固定的数据成员与校验信息
artifacts/                   本地训练、评测和预测产物（不提交）
```

## 开发与贡献

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv build
uv run twine check dist/*
```

测试使用本地合成数据，不会下载 Oxford 数据集或预训练权重。提交修改前请阅读
[贡献指南](CONTRIBUTING.zh-CN.md)；安全漏洞请按 [SECURITY.md](SECURITY.md) 私下报告，
不要创建公开 issue。项目采用 [MIT License](LICENSE)。

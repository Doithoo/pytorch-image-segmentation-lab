# 代码导览：先运行什么，再读哪里

[English](code-tour.md) | [返回 README](../README.zh-CN.md)

完整工程代码比课堂伪代码长，因为它还要处理错误、配置、保存和恢复。第一次不必顺序
读完 `src/`；沿一张图片从 manifest 到预测结果，更容易建立整体认识。

## 推荐阅读顺序

| 步骤 | 先运行 | 再阅读 | 重点 |
| --- | --- | --- | --- |
| 1 | `examples/01_mask_and_logits.py` | `data/dataset.py::map_mask` | 原始值怎样成为目标 |
| 2 | `examples/02_paired_transform.py` | `data/transforms.py` | 图片与掩码怎样保持对齐 |
| 3 | `examples/03_dataloader_batch.py` | `data/dataset.py` | 单个样本怎样组成 batch |
| 4 | `examples/04_minimal_training_loop.py` | `models/unet.py`、`training/losses.py` | logits 和 loss 从哪里来 |
| 5 | `segment train ... --dry-run` | `training/train.py`、`trainer.py` | 正式组件怎样组装 |
| 6 | `examples/05_checkpoint_overlay.py` | `inference/predictor.py` | checkpoint 怎样变成掩码 |

## 数据从一行 CSV 开始

`data/manifest.py` 扫描、校验并划分图片/掩码对，写出相对路径 CSV。Oxford 官方列表
和通用配对目录的差异只存在于准备阶段；训练时都交给 `SegmentationDataset`。

Dataset 每次读取一行，打开 RGB 图片和单通道掩码，执行同一组几何变换，再用 `map_mask`
把原始值映射到 0、1、255。这里值得逐行读，因为错误标签会影响后面所有模块。

## 模型只负责图片到 logits

`models/unet.py` 是主学习模型。输入 `[B,3,H,W]`，输出 `[B,2,H,W]`，不知道 CSV、
标签值或 artifact 目录。

`models/registry.py` 根据配置名称创建模型；`models/deeplab.py` 把 torchvision 的字典
输出转换成相同 logits 形状。第一遍可跳过 DeepLab，先完整读完 80 行 U-Net。

## loss 和 Trainer 怎样连接

`training/losses.py` 组合交叉熵与 foreground soft Dice，并过滤 ignore index。
`training/train.py` 负责从配置创建 Dataset、DataLoader、模型、optimizer、scheduler 和
运行目录；`training/trainer.py` 执行 epoch，并保存 best/last checkpoint。

```bash
uv run segment train --config configs/learning_minimal.yaml --dry-run
```

这条命令会经过上述所有模块一次。调试时从 CLI 入口向下追踪，比从几十个 helper 猜调用
关系更快。

## 评测与推理的区别

`evaluation/metrics.py` 只做计数和公式；`evaluation/evaluate.py` 加载有标签 Dataset，
写 JSON/CSV 并选择代表图；`evaluation/visualization.py` 把原始类别掩码与展示颜色分开。

`inference/predictor.py` 不需要目标掩码。它从 checkpoint 恢复模型与预处理，对图片预测，
再把类别索引还原到原图尺寸。

## 第一遍可以跳过

- `config.py` 的严格类型校验细节；先会用 `show-config`。
- `manifest.py` 的哈希、Oxford 审计辅助函数；先看 Dataset 读取一行。
- `checkpoint.py` 的 RNG 状态；先分清 best 与 last。
- `cli.py` 的 argparse 声明；它主要负责终端界面与错误转换。
- `evaluation/visualization.py` 的绘图像素细节；先读输出文件。

遇到配置覆盖问题再看[配置流](configuration-flow.zh-CN.md)，遇到公式问题再看
[系统原理](how-it-works.zh-CN.md)。

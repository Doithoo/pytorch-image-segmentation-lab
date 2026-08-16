# 可运行示例

[English](README.md) | [学习路径](../docs/learning-path.zh-CN.md)

请按编号依次运行。每个文件都刻意保持小巧，并在模块 docstring 中标明对应生产代码。

## 示例 01-02：像素含义与对齐

示例 01 映射 Oxford 标签、创建 logits、计算损失并预测类别索引。示例 02 强制水平
翻转，用来展示图像和掩码几何必须一起移动。

```bash
uv run python examples/01_mask_and_logits.py
uv run python examples/02_paired_transform.py
```

## 示例 03-04：批次与优化

示例 03 对照 `[B,C,H,W]` 图像和 `[B,H,W]` 掩码。示例 04 运行一次完整 U-Net
优化步骤，并展示损失分量。

```bash
uv run python examples/03_dataloader_batch.py
uv run python examples/04_minimal_training_loop.py
```

## 示例 05：Checkpoint 推理

该示例需要 checkpoint 和图像。它从 checkpoint 恢复模型构造和预处理，并保存
原始、彩色和叠加输出。

```bash
uv run python examples/05_checkpoint_overlay.py \
  --checkpoint artifacts/learning-unet/best.pt \
  --image data/raw/images/Abyssinian_1.jpg --device cpu
```

示例是教学入口，不是另一套实现。正式实验请使用 `segment` CLI。

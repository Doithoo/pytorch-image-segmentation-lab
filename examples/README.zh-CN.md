# 五个先运行的小程序

[English](README.md) | [实践路线](../docs/tutorial/learning-path.zh-CN.md)

这些文件把正式工程中的一件事单独拿出来。建议按编号运行，先猜输出，再对照代码。它们
不是另一套训练框架；每个文件顶部都指向对应的正式模块。

| 示例 | 运行后看什么 | 对应源码 |
| --- | --- | --- |
| `01_mask_and_logits.py` | `1/2/3 -> 1/0/255`、logits/target/prediction 形状 | `data/dataset.py`、`training/losses.py` |
| `02_paired_transform.py` | 强制翻转后红色区域和宠物 mask 是否仍对齐 | `data/transforms.py` |
| `03_dataloader_batch.py` | batch 图片有通道，mask 没有通道 | `data/dataset.py` |
| `04_minimal_training_loop.py` | CE、Dice、总 loss 和参数是否真的改变 | `models/unet.py`、`training/trainer.py` |
| `05_checkpoint_overlay.py` | checkpoint 怎样恢复模型并写三种 PNG | `inference/predictor.py` |

```bash
uv run python examples/01_mask_and_logits.py
uv run python examples/02_paired_transform.py
uv run python examples/03_dataloader_batch.py
uv run python examples/04_minimal_training_loop.py
```

前四个示例不下载数据。第五个需要你已经训练出的 checkpoint 和一张图片：

```bash
uv run python examples/05_checkpoint_overlay.py \
  --checkpoint artifacts/first-unet/best.pt \
  --image data/raw/images/Abyssinian_1.jpg \
  --output artifacts/example-prediction.png --device cpu
```

运行后应看到原始类别掩码、`.color.png` 和 `.overlay.png`。继续学习时使用 `segment`
CLI 完成正式训练和评测。

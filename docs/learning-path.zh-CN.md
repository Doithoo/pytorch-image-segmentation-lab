# 学习路径

[English](learning-path.md) | [返回 README](../README.zh-CN.md)

请按阶段依次学习。每一阶段先运行一个很小的示例，再进入生产实现。只有能够解释
前面阶段的张量形状和产物后，才开始耗时训练。

## 阶段 1：像素、掩码与 Logits

运行 `examples/01_mask_and_logits.py`。确认图像批次形状是 `[B, 3, H, W]`，
logits 是 `[B, 2, H, W]`，目标是 `[B, H, W]`。理解为何预测使用
`argmax(dim=1)`，而交叉熵直接接收 logits。

```bash
uv run python examples/01_mask_and_logits.py
```

然后阅读[教程 00](tutorial/00-basics.zh-CN.md)，并通过
[代码导览](code-tour.zh-CN.md)查看 `data/dataset.py`。

## 阶段 2：成对数据

运行示例 02 和 03。几何变换必须对图像和掩码作出相同的随机选择。图像使用双线性
插值，掩码使用最近邻插值，避免类别 ID 变成小数。

```bash
uv run python examples/02_paired_transform.py
uv run python examples/03_dataloader_batch.py
```

下载 Oxford、准备清单并在训练前检查预览。你应能解释原始 `1/2/3` 与映射后的
`1/0/255`。

## 阶段 3：一次优化步骤

运行示例 04，追踪前向、损失、梯度清零、反向和优化器更新。然后运行真实 dry-run。
它使用正式 Dataset、U-Net、组合损失和优化器，但不会创建常规实验目录。

```bash
uv run python examples/04_minimal_training_loop.py
uv run segment train --config configs/learning_minimal.yaml --dry-run
```

继续阅读[教程 03](tutorial/03-unet.zh-CN.md)和
[教程 04](tutorial/04-training.zh-CN.md)。

## 阶段 4：一个小实验

训练 CPU 入门配置。比较训练和验证损失，但要用验证集宠物 IoU 判断哪个 checkpoint
泛化更好。绝不能使用官方测试集调参。

```bash
uv run segment train --config configs/learning_minimal.yaml
```

改变变量前先写下问题。按照[实验指南](experiments.zh-CN.md)进行受控对照。

## 阶段 5：评估与推理

运行示例 05，评估最佳 checkpoint，并同时检查数值表格和叠加图。区分全局混淆
指标与逐图指标平均值。不要只看一个分数，还要观察假阳性和假阴性。

```bash
uv run python examples/05_checkpoint_overlay.py \
  --checkpoint artifacts/learning-unet/best.pt \
  --image data/raw/images/Abyssinian_1.jpg --device cpu
```

最后阅读[教程 05](tutorial/05-evaluation-and-inference.zh-CN.md)，再通过
[使用自己的数据](using-your-data.zh-CN.md)改造项目。

## 完成检查

当你能解释标签 3、成对变换、logits 与掩码、忽略像素、宠物 IoU 与 Dice、验证集
与测试集、checkpoint 内容以及一个可视化失败案例时，就可以开始扩展项目。

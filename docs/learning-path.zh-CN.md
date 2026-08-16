# 图像分割学习路线

[English](learning-path.md) | [返回 README](../README.zh-CN.md)

这条路线面向会使用基本 Tensor 和 loss、但第一次做像素级预测的学习者。完整走一遍约
需要 **8–12 小时**；其中大部分时间花在观察数据、阅读短代码和解释结果，而不是等待
训练。

第一次不必记住所有公式。先让下面这条链路真的运行起来：

```text
图片/掩码 -> Dataset -> U-Net logits -> loss -> checkpoint -> IoU/Dice -> 错误叠加图
```

## 开始前：确认环境（约 20 分钟）

```bash
uv sync --extra dev
uv run segment --version
uv run segment show-config --config configs/learning_minimal.yaml
```

最终配置应显示 128 像素、最多 256 个训练样本、两轮、小型 U-Net 和 `device: auto`。
`show-config` 比凭记忆判断更可靠：后面每次使用 `--set` 后，都可以先回来看最终值。

环境或设备异常时先读[教程 01](tutorial/01-environment.zh-CN.md)，不要在安装尚未稳定时
同时更换 PyTorch、CUDA、模型和数据。

## 阶段 1：看懂四种张量（约 45 分钟）

```bash
uv run python examples/01_mask_and_logits.py
uv run python examples/03_dataloader_batch.py
```

你会看到：

```text
单张图像       [3, H, W]
图像批次       [B, 3, H, W]
整数目标       [B, H, W]
模型 logits    [B, 2, H, W]
预测类别       [B, H, W]
```

目标没有通道维度，因为每个像素只保存一个类别 ID。模型输出有两个通道，因为每个像素
都要给背景和宠物各一个分数。交叉熵直接读取 logits；`argmax(dim=1)` 只在决定预测类别
时使用。

完成标准：你能解释为什么不能在计算交叉熵之前先 `argmax`。不确定时读
[教程 00](tutorial/00-basics.zh-CN.md)。

## 阶段 2：检查数据和掩码（约 1.5 小时）

```bash
uv run python scripts/download_data.py --data-dir data/raw
uv run segment prepare-data --config configs/learning_minimal.yaml \
  --data-dir data/raw --manifest-dir data/manifests \
  --source-format oxford-pet
uv run python scripts/preview_dataset.py data/manifests/train.csv \
  --output artifacts/dataset_preview.png --limit 4
uv run python examples/02_paired_transform.py
```

打开预览图，不要只确认“文件生成了”。逐行检查：图片和原始 trimap 是否对应；映射后的
绿色宠物区域是否正确；灰色忽略带是否没有变成第三类；叠加边缘是否对齐。

这里最容易犯两个错误：只翻转图片而没有翻转掩码，以及用双线性插值缩放类别掩码。
前者让标签错位，后者会创造不存在的小数类别。示例 02 用一个左右分区的合成样本把这
件事变得可见。

完成标准：你能从原始 `1/2/3` 解释映射后的 `1/0/255`，并知道 255 在哪里被忽略。
继续阅读[教程 02](tutorial/02-data-and-masks.zh-CN.md)。

## 阶段 3：读懂一次参数更新（约 1.5 小时）

```bash
uv run python examples/04_minimal_training_loop.py
uv run segment train --config configs/learning_minimal.yaml --dry-run
```

沿下面顺序读示例，不要先钻进完整 Trainer：

```text
optimizer.zero_grad()
logits = model(images)
loss = criterion(logits, targets)
loss.backward()
optimizer.step()
```

再打开 `src/image_segmenter/models/unet.py`，从输入追踪四次下采样、bottleneck、四个
skip connection 和最终 `1x1` 卷积。模型只看图片，从不读取目标掩码；掩码只在 loss
和指标中出现。

`dry-run OK` 说明正式 Dataset、U-Net、loss 和 optimizer 已经连通。它只更新一个
batch，不创建完整运行目录。

完成标准：你能指出哪一步计算梯度，哪一步真正修改参数。模型结构见
[教程 03](tutorial/03-unet.zh-CN.md)，训练细节见[教程 04](tutorial/04-training.zh-CN.md)。

## 阶段 4：完成第一个小实验（约 1–2 小时）

```bash
uv run segment train --config configs/learning_minimal.yaml \
  --set run_name first-unet
```

训练结束后依次打开：

1. `config.yaml`：确认最终配置，不要只相信命令历史。
2. `metrics.csv`：比较 train loss、valid loss 和 validation pet IoU。
3. `best.pt` 与 `last.pt`：确认最佳验证状态不一定是最后一轮。
4. `run.yaml`：记录设备、版本、revision 和 manifest 哈希。

两轮结果不一定很好。这个阶段的目标是得到一套可以检查的真实产物，而不是追逐高分。
训练 loss 降低但验证 IoU 不变，可能只是模型更适应了训练样本；先看预测，再决定是否
需要更多 epoch、Dice loss 或更宽的模型。

完成标准：你能用 `metrics.csv` 说明为什么选择某个 checkpoint，而不是只说“它分数
最高”。

## 阶段 5：从数字回到图片（约 1.5 小时）

```bash
uv run segment evaluate --checkpoint artifacts/first-unet/best.pt \
  --split test --plot --device cpu
uv run python examples/05_checkpoint_overlay.py \
  --checkpoint artifacts/first-unet/best.pt \
  --image data/raw/images/Abyssinian_1.jpg --device cpu
```

先比较 pet IoU、pet Dice 和 pixel accuracy，再看 `sample_overlays/` 与 `worst_cases/`。
如果 accuracy 很高但宠物轮廓明显缺失，说明大面积背景掩盖了前景错误。全局 IoU 把
整个数据集像素汇总后计算；逐图 IoU 先算每张图再平均，两者回答的问题不同。

完成标准：从一张错误叠加图中指出假阳性、假阴性和忽略区域。详见
[教程 05](tutorial/05-evaluation-and-inference.zh-CN.md)。

## 阶段 6：做一个可解释的对比（约 2–4 小时）

先写问题，再运行实验。推荐从 CE 与 CE+Dice 开始：

```bash
uv run segment train --config configs/learning_minimal.yaml \
  --set run_name exp-ce
uv run segment train --config configs/learning_minimal.yaml \
  --set loss.cross_entropy_weight 0.5 \
  --set loss.dice_weight 0.5 \
  --set run_name exp-ce-dice
```

保持 manifest、seed、U-Net 宽度、样本上限和 epoch 相同。比较最佳验证宠物 IoU、曲线
和同一批图片上的错误类型。只有两轮和一个 seed 时，结论应写成“这次运行中发生了
什么”，不要写成 Dice 对所有数据都更好。

更多对照见[实验指南](experiments.zh-CN.md)。准备换自己的数据时读
[自己的数据指南](using-your-data.zh-CN.md)。

## 完成后你应该能回答

- 为什么掩码缩放必须用最近邻？
- 为什么模型输出 `[B,2,H,W]`，目标却是 `[B,H,W]`？
- 原始值 3、训练索引 255 和模型类别有什么区别？
- `best.pt` 与 `last.pt` 各自表示什么？
- 为什么测试集不能用于选择 loss 权重？
- 为什么 pixel accuracy 不能替代宠物 IoU？
- 一张最差案例图能告诉你哪些汇总分数看不到的问题？

不必一次回答得完美。能够回到数据、代码和产物找到证据，比背下一组术语更重要。

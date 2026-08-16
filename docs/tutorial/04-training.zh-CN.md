# 教程 04：从一个 batch 到可恢复实验

[English](04-training.md) | [教程索引](README.zh-CN.md)

训练循环的核心只有五步。真实工程增加验证、日志和 checkpoint，不会改变这五步的
因果顺序。

## 先读最小循环

```bash
uv run python examples/04_minimal_training_loop.py
```

对应代码是：

```python
optimizer.zero_grad(set_to_none=True)
logits = model(images)
loss, components = criterion(logits, targets)
loss.backward()
optimizer.step()
```

- `zero_grad` 清除上一个 batch 的梯度；PyTorch 默认会累加梯度。
- forward 根据当前参数产生 logits。
- loss 比较 logits 与目标，为每个参数建立可导的错误信号。
- backward 计算梯度，但还没有修改参数。
- optimizer.step 才真正更新参数。

想验证最后一点，可以比较某一层在 step 前后的权重。增强后的示例会直接打印参数变化，
比只看一个随机 loss 更容易确认更新确实发生。

## cross-entropy 与 soft Dice

交叉熵逐像素判断类别，容易理解且数值稳定。当前景面积远小于背景时，soft Dice 直接
关注预测前景与目标前景的重叠。

组合 loss 是：

```text
total = ce_weight * cross_entropy + dice_weight * soft_dice
```

`learning_minimal.yaml` 先只用 CE，让第一条训练路径保持简单；`reference_unet.yaml`
使用 0.5 CE + 0.5 Dice。权重总和不必严格等于 1，但两者不能同时为 0。

忽略像素必须在两个 loss 中以相同方式排除。若 CE 忽略 255 而 Dice 仍计算它，模型会
收到互相矛盾的训练信号。

## dry run 检查什么

```bash
uv run segment train --config configs/learning_minimal.yaml --dry-run
```

它从正式 train manifest 加载一个 batch，执行 U-Net、loss、backward 和 step，并打印
设备、形状、mask values、有效像素比例和 loss。成功时最后一行是 `dry-run OK`。

它不会创建常规运行目录，也不报告精度。dry run 的问题是“管道能否优化一个 batch”，
不是“模型是否泛化”。

## 一个 epoch 发生什么

Trainer 对训练集和验证集执行相似循环，但只有训练阶段启用梯度和参数更新：

```text
train:
  model.train()
  forward -> loss -> backward -> optimizer.step

valid:
  model.eval()
  no_grad()
  forward -> loss -> metrics
```

每轮结束后记录 train loss、train pet IoU、valid loss、valid pet IoU 和学习率。验证集
不更新模型，它用于比较 epoch 和配置。

## 怎样读第一条曲线

```bash
uv run segment train --config configs/learning_minimal.yaml \
  --set run_name first-unet
uv run python scripts/plot_metrics.py artifacts/first-unet/metrics.csv \
  --output artifacts/first-unet/loss_curve.png
```

| 现象 | 可能原因 | 先做什么 |
| --- | --- | --- |
| train/valid loss 都下降 | 模型正在学习 | 继续看 IoU 和预测图 |
| train loss 降，valid loss 升 | 可能过拟合 | 看错误图，再尝试增强/较小模型 |
| 两者几乎不变 | lr、数据或容量问题 | 回到 preview 与 dry run，再检查 lr |
| loss 降但 pet IoU 为 0 | 模型偏向背景 | 看前景比例，尝试 CE+Dice |
| loss NaN | 无有效像素或优化不稳定 | 检查 mask values/valid ratio，降低 lr |

两轮曲线点太少，不适合判断长期过拟合。它的价值是让读取产物成为习惯。

## best.pt 与 last.pt

`last.pt` 保存最近完成的 epoch，适合中断后继续训练。`best.pt` 保存验证 pet IoU 最高
的状态，适合最终评测和推理。若后期过拟合，两者会不同。

checkpoint 不只包含模型权重，还包含：

- optimizer 和 scheduler 状态；
- 完整配置与模型元数据；
- 标签映射和 ignore index；
- manifest 哈希；
- Python、NumPy、PyTorch 及可用设备随机状态；
- epoch、最佳分数、包版本和 Git revision。

它使用 Python/PyTorch 序列化，只加载自己生成或信任来源的 checkpoint。

## 怎样安全续训

```bash
uv run segment train --config configs/reference_unet.yaml \
  --resume artifacts/reference-unet/last.pt
```

恢复前会比较模型、预处理、loss、batch、标签和 manifest。改变这些训练语义会被拒绝，
避免把两个不同实验拼成一条曲线。

`scheduler: none` 时可以增加 `train.epochs`，继续多跑几轮。cosine scheduler 的
`T_max` 由计划总轮数决定；checkpoint 已经保存旧调度状态，因此恢复时必须保持原来的
总 epoch。想延长 cosine 实验，应创建新的 run 并明确说明新的调度计划，而不是假装与
原计划逐步等价。

## seed 能保证什么

项目为 Python、NumPy、PyTorch 和 DataLoader worker 设置 seed，并保存恢复所需随机
状态。这可以让同设备、同版本、同 manifest 的中断恢复保持一致。

seed 不能消除所有硬件/库差异，也不能让单次实验代表普遍规律。需要可靠结论时应运行
多个 seed，报告均值与波动；学习阶段至少要把“只跑一个 seed”写进限制。

## 第一个受控实验

比较 CE 与 CE+Dice：

```bash
uv run segment train --config configs/learning_minimal.yaml \
  --set run_name exp-ce
uv run segment train --config configs/learning_minimal.yaml \
  --set loss.cross_entropy_weight 0.5 \
  --set loss.dice_weight 0.5 \
  --set run_name exp-ce-dice
```

先猜测哪组更快改善宠物 IoU，再比较验证指标和同一批叠加图。不要因为某组测试集分数
更高就继续调它；那会把 test 变成 validation。

训练完成后进入[教程 05：评测与推理](05-evaluation-and-inference.zh-CN.md)。异常现象可
直接查[排错指南](../troubleshooting.zh-CN.md)。

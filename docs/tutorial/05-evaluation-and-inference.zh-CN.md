# 教程 05：别让一个分数替你看图片

[English](05-evaluation-and-inference.md) | [教程索引](README.zh-CN.md)

评测使用有标签数据回答“模型表现怎样”；推理对没有标签的新图片生成掩码。两者都从
checkpoint 恢复模型和预处理，但输出目的不同。

## 运行完整评测

```bash
uv run segment evaluate --checkpoint artifacts/first-unet/best.pt \
  --split test --plot --device cpu
```

默认输出在 checkpoint 同级的 `evaluation/`：

```text
metrics.json                  汇总指标和样本数
per_class_metrics.csv         背景/宠物各自 IoU 与 Dice
per_image_metrics.csv         每张图片的宠物指标和前景比例
sample_overlays/              固定顺序的代表样本
worst_cases/                  按逐图 pet IoU 排出的较差样本
loss_curve.png                可选训练/验证 loss 曲线
```

先读数字，再打开图片。只做其中一半都会丢失信息。

## 从混淆计数理解 IoU

对宠物前景：

```text
TP：目标是宠物，预测也是宠物
FP：目标是背景，预测成宠物
FN：目标是宠物，预测成背景
```

```text
IoU  = TP / (TP + FP + FN)
Dice = 2TP / (2TP + FP + FN)
```

Dice 对同一预测通常高于 IoU，但两者排序趋势相近。报告时写清具体指标，不要把 0.90
Dice 与 0.90 IoU 当成同一个质量水平。

pixel accuracy 还会计入大量容易的背景像素，因此不能代替前景指标。

## 全局指标与逐图均值

全局 pet IoU 先汇总整个数据集的 TP/FP/FN，再计算一次；大图和前景多的图贡献更多。
逐图平均为每张图各算一次 IoU，再求均值，让每张图权重接近相同。

两者不同并不表示实现错误。它们分别回答：

- 全局：所有有效像素合在一起，重叠质量怎样？
- 逐图：一张典型图片的重叠质量怎样？

参考运行同时报告两者，避免只选择更好看的数字。

## 没有前景时怎么办

若目标和预测都没有宠物，分子和分母都为 0，逐图 pet IoU/Dice 没有定义。项目会把这
类值写为空并计数，从逐图均值中排除，而不是当作满分。

如果目标为空但模型预测出宠物，IoU 是 0；这仍是可诊断的假阳性。参考 Oxford test 中
存在少数没有宠物前景的上游 mask，因此必须明确处理。

## 读一张错误叠加图

![目标、预测和像素错误](../assets/reference-unet-overlay.png)

四列分别是原图、目标、预测和错误叠加。错误图使用不同颜色区分真阳性、假阳性、
假阴性与忽略区域。观察时按顺序问：

1. 错误集中在轮廓还是整块主体？
2. 细腿、尾巴、耳尖是否被漏掉？
3. 背景纹理是否被误认为宠物？
4. 目标本身是否异常、反转或错位？
5. 网络拉伸是否让物体比例改变得太多？

最差案例不一定都是模型失败，也可能暴露上游标注问题。不要为了提高数字就悄悄删除
困难样本；先记录规则，再决定数据清理是否适用于所有实验。

## 为什么 official test 只留到最后

验证集用于选择 checkpoint、loss 权重和模型宽度。若每次尝试都查看 test 分数，再选择
最高的一组，test 已经参与调参，不能再代表未见数据。

学习阶段可以反复在 validation 上做短对比。确定最终设置后，再运行一次完整 test 评测，
记录配置、manifest、环境和限制。

## 对新图片推理

```bash
uv run segment predict --checkpoint artifacts/first-unet/best.pt \
  --image data/raw/images/Abyssinian_1.jpg \
  --output artifacts/predictions/Abyssinian_1.png --device cpu
```

输出包括：

- `Abyssinian_1.png`：单通道类别索引 0/1，可供程序读取；
- `Abyssinian_1.color.png`：人眼查看的颜色；
- `Abyssinian_1.overlay.png`：预测区域叠加到原图。

Predictor 从 checkpoint 恢复网络尺寸、归一化和模型结构，把网络预测用最近邻插值还原
到原图尺寸。彩色和 overlay 文件只用于展示，不能重新作为目标掩码。

目录推理使用相同命令，把 `--image` 指向目录、`--output` 指向另一个目录。输出目录
不能位于输入目录内部，防止递归读取自己刚生成的图片。

## 一次完整的结果解释

不要只写“pet IoU = 0.90”。至少包含：

- 使用哪个 manifest split 和 checkpoint；
- 全局 pet IoU/Dice 与逐图均值；
- 样本数、空目标数和 ignored pixels；
- 一张代表性成功图和一张失败图；
- 配置、设备、版本和一个已知限制。

接下来可以按[实验指南](../experiments.zh-CN.md)做受控对比，或按
[自己的数据指南](../using-your-data.zh-CN.md)迁移到新任务。

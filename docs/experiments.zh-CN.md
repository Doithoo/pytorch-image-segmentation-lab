# 可复现分割实验指南

[English](experiments.md) | [返回 README](../README.zh-CN.md)

一个实验只回答一个写下来的问题。先得到基线，再改变一个因素；保存配置和图片，比在
终端里记住一串分数更可靠。

## 运行前先写五行

```text
问题：我想知道什么？
假设：我预计哪一组会怎样变化？
保持不变：manifest、seed、模型、epoch、预处理……
唯一变化：哪个配置字段？
判断依据：哪个 validation 指标和哪些图片？
```

测试集不出现在“判断依据”中。它只用于选定最终设置后的报告。

## 建立可检查的基线

```bash
uv run segment show-config --config configs/learning_minimal.yaml
uv run segment train --config configs/learning_minimal.yaml \
  --set run_name baseline-ce
```

保留 `config.yaml`、`metrics.csv`、`run.yaml`、best/last checkpoint。两轮基线可能很弱，
但仍能证明数据、训练和评测路径是否正常。不要通过同时改五个参数来“救”一条还没看懂
的曲线。

## 对比 1：CE 与 CE+Dice

问题：在相同短训练中，加入 foreground Dice 是否改善 validation pet IoU？

```bash
uv run segment train --config configs/learning_minimal.yaml \
  --set run_name exp-ce
uv run segment train --config configs/learning_minimal.yaml \
  --set loss.cross_entropy_weight 0.5 \
  --set loss.dice_weight 0.5 \
  --set run_name exp-ce-dice
```

比较最佳 validation pet IoU、valid loss，以及同一批图上的漏检/误检。组合 loss 的数值
尺度与纯 CE 不同，不能只比较 loss 数字大小。

## 对比 2：U-Net 宽度

问题：更宽的 U-Net 是否值得额外时间和内存？

```bash
uv run segment train --config configs/learning_minimal.yaml \
  --set model.base_channels 8 --set run_name exp-width-8
uv run segment train --config configs/learning_minimal.yaml \
  --set model.base_channels 16 --set run_name exp-width-16
```

记录每轮时间、峰值内存（若设备可查看）、参数量和 validation pet IoU。宽度翻倍时参数
不会只翻倍；小数据也可能让更宽模型更快过拟合。

## 对比 3：水平翻转

问题：左右翻转是否帮助 Oxford 宠物泛化？

```bash
uv run segment train --config configs/learning_minimal.yaml \
  --set data.horizontal_flip 0.0 --set run_name exp-no-flip
uv run segment train --config configs/learning_minimal.yaml \
  --set data.horizontal_flip 0.5 --set run_name exp-flip
```

几何增强必须同时作用于图片和掩码。若任务包含文字方向或具有固定左右语义，翻转可能
不合理；增强是否有效取决于数据含义，不是默认真理。

## 对比 4：U-Net 与 DeepLabV3

```bash
uv run segment train --config configs/reference_unet.yaml \
  --set run_name compare-unet
uv run segment train --config configs/deeplabv3_resnet50.yaml \
  --set run_name compare-deeplab
```

这不是纯架构对比：默认 DeepLab 使用 ImageNet 预训练 ResNet50，U-Net 从随机初始化
开始，batch size、学习率和训练轮数也不同。它回答的是“两套完整方案表现怎样”，不能
证明 DeepLab 架构本身优于 U-Net。

若要更接近单变量对比，应统一输入尺寸、训练轮数、loss 和预训练条件，并说明不同模型
可能仍需要不同最优超参数。

## 怎样比较结果

建议为每组建立一行表格：

| run | 唯一变化 | best valid pet IoU | valid loss | 时间 | 观察到的错误 |
| --- | --- | ---: | ---: | ---: | --- |
| baseline | CE | … | … | … | … |
| candidate | CE+Dice | … | … | … | … |

再放两张使用相同 sample ID 的叠加图。数字告诉你差多少，图片告诉你差在边界、细结构、
背景误报还是标注异常。

## 常见无效对比

- 两组使用不同 manifest 或样本上限；
- 改了模型的同时改变预训练、loss、输入尺寸和 epoch；
- 只保留最好的一次随机 seed；
- 根据 test 分数选择设置；
- 比较 loss 数值却忽略 loss 定义不同；
- 不记录运行时间、设备或失败的 run；
- 从一个数据集和一个 seed 推导普遍结论。

## 报告模板

```text
问题与假设：
数据/manifest SHA-256：
基线配置与候选配置：
保持不变的设置：
validation 结果：
最终 test 结果（若已选定方案）：
代表成功与失败案例：
运行环境与耗时：
限制和下一步：
```

[参考运行](reference-run.zh-CN.md)展示一份完整记录。先用短配置学习比较方法，再决定
是否值得运行 25 epoch 或多个 seed。

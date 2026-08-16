# 图像分割实操教程

[English](README.md) | [返回 README](../../README.zh-CN.md)

不必一次读完六章。先跑 README 中的最短流程，遇到哪个问题，再打开对应章节。

| 章节 | 解决什么问题 | 什么时候读 |
| --- | --- | --- |
| [00 分割基础](00-basics.zh-CN.md) | logits、掩码和交叉熵是什么 | 张量形状还不熟时 |
| [01 环境与设备](01-environment.zh-CN.md) | 怎样安装、选设备和判断速度是否正常 | 命令跑不起来时 |
| [02 数据与掩码](02-data-and-masks.zh-CN.md) | 原始 trimap 怎样变成训练目标 | 准备数据或换数据时 |
| [03 U-Net](03-unet.zh-CN.md) | 编码器、skip connection 和解码器怎样连接 | 准备读模型代码时 |
| [04 训练](04-training.zh-CN.md) | loss、优化器、checkpoint 和续训怎样配合 | 开始正式实验前 |
| [05 评测与推理](05-evaluation-and-inference.zh-CN.md) | IoU/Dice 怎么看，错误图怎么读 | 模型训练完成后 |

推荐顺序是 `00 -> 02 -> 03 -> 04 -> 05`。环境已经正常时，01 可以作为随手查询的
手册。每章最后都有一个可以实际回答的问题；先自己写下答案，再回到代码验证。

遇到具体报错时直接查看[排错指南](../troubleshooting.zh-CN.md)。

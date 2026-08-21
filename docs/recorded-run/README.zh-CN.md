# 经过核对的 U-Net 参考运行

[English](README.md) | [返回文档导航](../README.zh-CN.md)

这不是一个要求每台机器复现到最后小数位的榜单分数，而是一份可以逐项检查的完整运行。
配置、manifest、训练曲线、验证选择和 test 指标来自同一次实验。

## 为什么保留这次运行

初学者第一次看到 `0.90 IoU` 时，容易忽略它对应哪份数据、哪个 checkpoint、怎样选择
以及哪些样本仍然失败。本页把数字和产物放回同一上下文，也诚实记录一次中断恢复。

## 数据与协议

- 数据：固定 SHA-256 的 Oxford-IIIT Pet 官方图片与标注归档。
- 划分：官方 3,669 张 test 不变；trainval 在每个品种内按 seed 666 切成 2,941 train、739 valid。
- 标签：原始 1=pet、2=background、3=ignore 255。
- 模型：`unet_small`，base channels 32，256x256 输入。
- loss：0.5 cross-entropy + 0.5 foreground Dice。
- 选择：25 轮中 validation pet IoU 最高的 checkpoint。
- test 使用：选定 checkpoint 后完整评测一次。

三个 manifest SHA-256：

```text
train f7e7af12141c01b6297c6bbc935a536a90023f80f1c57297e5477e4b7cd67371
valid 872ed4b206b576a1c0582c7ec8add1b782e2cff68624ebd8901c6b64a1a5897d
test  619642b927ecd138e3bc068340b5c30231ea705e48d27420cc9d2e20034b757c
```

## 配置和环境

[最终配置](reference-unet-config.yaml)使用 AdamW、lr `3e-4`、weight decay
`1e-4`、cosine scheduler、batch 8、MPS 设备。

[运行元数据](reference-unet-run.yaml)记录 macOS 26.5.2 arm64、Python
3.11.15、PyTorch 2.13.0、torchvision 0.28.0，以及代码 revision
`1deff1a03ba2d4549c538b4eef36df3868abe92b`。

端到端训练耗时 19,253 秒（约 5 小时 21 分）。第 15 轮开始前触发 MPS 高水位内存限制，
从第 14 轮 `last.pt` 恢复后完成。记录 revision 当时没有保存独立 DataLoader generator，
因此这次运行不宣称与未中断训练逐比特一致；后续修复已经消除该缺口。

## 验证与测试结果

最佳 checkpoint 来自第 24 轮，validation pet IoU 为 `0.902836`。

| 指标 | test 实测值 |
| --- | ---: |
| Pet IoU（全局） | 0.906669 |
| Pet Dice（全局） | 0.951050 |
| Background IoU（全局） | 0.950423 |
| Mean IoU（全局） | 0.928546 |
| Pixel accuracy | 0.966539 |
| Pet IoU（逐图均值） | 0.905472 |
| 有效像素 | 210,899,866 |
| 忽略像素 | 29,551,718 |
| 空目标样本 | 7 |

机器可读结果见[评测 JSON](reference-unet-evaluation.json)、
[逐类表格](reference-unet-per-class.csv)、
[逐图表格](reference-unet-per-image.csv)和
[25 轮训练指标](reference-unet-training-metrics.csv)。

![训练和验证 loss](../assets/reference-unet-loss-curve.png)

训练 loss 持续下降，valid loss 在后期进入平台并有小幅波动。这比一条完美平滑曲线更
接近真实训练，也说明为什么要按 validation 指标保存 best，而不是默认使用最后一轮。

## 图片透露了什么

![代表性目标、预测和错误](../assets/reference-unet-overlay.png)

代表样本显示主体大区域已经学到，主要误差集中在边缘和忽略带附近。最差案例中还能
看到细结构漏检、背景误报，以及少数上游 trimap 的反转或空间不一致。

![一个低 IoU 案例](../assets/reference-unet-worst-case.png)

这些上游异常没有为了提高分数而移除。七张 test mask 不含宠物前景，其逐图前景指标
按文档规则处理。结果因此是对固定官方列表的报告，而不是清理后的理想子集。

## 为什么当前仓库树不保留 checkpoint

该模型文件约 89 MB，会增加仓库体积，而且 Git 不适合分发可替换的训练二进制，因此
当前仓库树不再保留它。记录的 checkpoint SHA-256 是：

```text
5291b96d17df4c197e46ab5ff4f3d9b36b58b0aba63d334f314681bebc3f9e16
```

仓库保留配置、元数据、指标和精选图片。该历史文件早于当前 weights-only checkpoint schema，
不要直接交给当前 CLI；只能在隔离且可信的旧环境中迁移，或重新运行以获得当前安全格式。
发布历史 checkpoint 时应使用 GitHub Release 或其他模型制品存储，并附上上述哈希；本地
重新运行：

```bash
uv run python scripts/download_data.py --data-dir data/raw
uv run segment prepare-data --config configs/reference_unet.yaml \
  --data-dir data/raw --manifest-dir data/manifests \
  --source-format oxford-pet
uv run segment train --config configs/reference_unet.yaml
uv run segment evaluate --checkpoint artifacts/reference-unet/best.pt \
  --split test --plot
```

新运行应比较 manifest 哈希、最终配置、环境和趋势。硬件或库不同带来的小数差异不表示
复现失败；数据成员或评测协议不同则不能直接比较。

# 排错指南

[English](troubleshooting.md) | [返回 README](../README.zh-CN.md)

先保留完整错误信息，再从最小命令开始。不要一边换设备、一边改模型、一边改数据；一次
只排除一个原因。

## 安装与命令

| 现象 | 常见原因 | 先做什么 |
| --- | --- | --- |
| `segment: command not found` | 没有通过 uv 运行，或项目未安装 | 运行 `uv sync --extra dev`，再用 `uv run segment --help` |
| Python 版本不支持 | 当前解释器不在 3.10–3.12 | 运行 `uv python pin 3.12` 后重新 `uv sync` |
| CUDA/MPS unavailable | 配置强制了本机没有的设备 | 先加 `--device cpu`，确认流程后再检查驱动 |
| import 来自错误目录 | 使用了另一个环境中的旧包 | 运行 `uv run python -c "import image_segmenter; print(image_segmenter.__file__)"` |

## 数据与掩码

| 现象 | 常见原因 | 先做什么 |
| --- | --- | --- |
| `missing image` / `missing mask` | 图片和掩码主干不同 | 对照 `images/a.jpg` 与 `masks/a.png` |
| `unexpected mask values` | `label_map` 或 `ignore_values` 未覆盖原始值 | 用 Pillow/NumPy 打印一张掩码的唯一值 |
| `empty split` | 数据太少或比例产生了 0 个样本 | 增加样本，或调整 `data.split_ratios` |
| 预览中边缘错位 | 图片与掩码尺寸不同，或几何变换没有成对执行 | 回到原始文件检查尺寸，再运行示例 02 |
| 掩码出现小数/新类别 | 对掩码使用了双线性插值 | 掩码缩放必须使用最近邻插值 |

## 训练

| 现象 | 常见原因 | 先做什么 |
| --- | --- | --- |
| dry run 失败 | 数据、标签、模型输出或损失没有连通 | 用 `uv run segment --debug train ...` 查看第一个异常 |
| loss 是 NaN | 学习率过大，或一个 batch 全是忽略像素 | 查看 mask values 和 valid pixel ratio，再降低 lr |
| loss 下降但 IoU 不涨 | 模型偏向背景，或验证样本太少 | 看前景比例和预测叠加图，再比较 CE 与 CE+Dice |
| 训练越来越好、验证变差 | 过拟合 | 保留基线，尝试翻转、较窄模型或更早停止 |
| MPS/CUDA OOM | batch、输入或模型太大 | batch 减半；仍失败时降低 image size/base channels |
| cosine 续训拒绝新 epochs | checkpoint 的调度计划与新总轮数不一致 | 保持原总轮数续训，或用新 run 重新规划训练 |

## 评测与推理

| 现象 | 常见原因 | 先做什么 |
| --- | --- | --- |
| pixel accuracy 高但图很差 | 背景像素占多数 | 看 pet IoU、Dice 和错误叠加图 |
| 找不到 manifest | checkpoint 保存的是相对 manifest 目录 | 从仓库根目录运行，或恢复对应的 `data/manifests` |
| 输出尺寸不对 | 网络尺寸与原图尺寸混淆 | 使用 predictor 生成输出，它会还原原图尺寸 |
| 彩色图不能重新训练 | `.color.png`/`.overlay.png` 是展示产物 | 训练只读取原始单通道类别掩码 |

## 仍然无法定位时

收集下面四项再开 issue：完整命令、第一个错误、`uv run segment --version` 输出、
`uv run segment show-config ...` 的最终配置。涉及数据时再附一张去除隐私的预览图和
掩码唯一值，不要上传整个数据集。

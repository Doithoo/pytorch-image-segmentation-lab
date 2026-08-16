# 教程 01：把环境跑稳，再谈加速

[English](01-environment.md) | [教程索引](README.zh-CN.md)

环境配置的目标不是“识别到最强显卡”，而是得到一个可重复运行、出错时知道从哪里查的
Python 环境。第一次建议用 CPU 完成 dry run，再决定是否切换 CUDA 或 Apple MPS。

## 安装锁定环境

项目支持 Python 3.10–3.12。在仓库根目录运行：

```bash
uv sync --extra dev
uv run segment --version
uv run segment --help
```

`uv sync` 根据 `uv.lock` 创建 `.venv` 并安装项目。`uv run` 会自动使用这个环境，通常
不需要手动激活。看到类似下面的版本输出，说明 CLI 入口已经安装：

```text
image-segmenter 0.1.0
```

如果终端直接输入 `segment` 找不到命令，而 `uv run segment` 正常，这是环境没有激活，
不是项目安装失败。学习阶段统一使用 `uv run ...` 最省心。

## 确认 Python 和 PyTorch 来自同一环境

```bash
uv run python -c "import sys, torch, image_segmenter; print(sys.version); print(torch.__version__); print(image_segmenter.__file__)"
```

`image_segmenter.__file__` 应指向当前仓库的 `src/image_segmenter`。如果指向别的目录，
说明你正在使用另一个环境中的旧安装。

## 看看本机有哪些设备

```bash
uv run python -c "import torch; print('cuda', torch.cuda.is_available()); print('mps', torch.backends.mps.is_available())"
```

`device: auto` 按 CUDA、MPS、CPU 的顺序选择。也可以在训练命令中明确指定：

```bash
uv run segment train --config configs/learning_minimal.yaml --dry-run --device cpu
```

显式请求不可用的 `cuda` 或 `mps` 会立即报错，而不是暗中退回 CPU。这样日志中的设备
信息才可信。

## 第一次为什么从 CPU 开始

最小配置限制了样本数、输入尺寸和模型宽度，CPU 可以完成 dry run。先在 CPU 验证数据
与代码，再切换加速器，可以把“数据错误”和“驱动/设备错误”分开。

不同设备的浮点实现可能产生轻微数值差异。固定 seed 提高可比性，但不保证 CUDA、MPS
和 CPU 逐比特相同。学习实验应比较趋势，并记录设备和版本。

## 先检查最终配置

```bash
uv run segment show-config --config configs/learning_minimal.yaml
```

重点确认：

- `image_size: 128`；
- `max_train_samples: 256`、`max_valid_samples: 64`；
- `epochs: 2`、`batch_size: 8`；
- `model.name: unet_small`；
- `pretrained: false`，不会下载权重。

配置正确后运行：

```bash
uv run python examples/01_mask_and_logits.py
```

这个示例不需要数据下载。它能运行，说明 Python、PyTorch 和项目包已经连通；后续下载
或 manifest 失败就属于数据阶段，而不是基础安装阶段。

## 常见设备问题

| 现象 | 先做什么 |
| --- | --- |
| CUDA 不可用 | 先用 CPU；再核对 NVIDIA 驱动和所装 PyTorch 构建 |
| MPS 运行中内存不足 | batch size 减半，必要时降低 image size/base channels |
| 多 worker 卡住 | 将 `data.num_workers` 设为 0，确认主流程后再增加 |
| 第一次运行很慢 | PyTorch 可能在初始化内核；比较完整 epoch，不只比较第一个 batch |
| 不同设备结果略有差异 | 记录版本、seed、manifest 和设备，比较趋势而非最后几位小数 |

不要通过关闭系统内存安全限制解决 OOM。减小 batch 或输入更容易复现，也更适合作为
初学者实验。

仍无法定位时查看[排错指南](../troubleshooting.zh-CN.md)。环境稳定后进入
[教程 02：数据与掩码](02-data-and-masks.zh-CN.md)。

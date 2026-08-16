# 教程 01：环境

[English](01-environment.md) | [教程索引](../learning-path.zh-CN.md)

项目使用 uv 固定并安装一个可复现 Python 环境。

## 安装

使用 Python 3.10-3.12。在仓库根目录运行：

```bash
uv sync --extra dev
uv run segment --version
uv run segment --help
```

`uv run` 在 `.venv` 内执行，手动激活并非必需。dev extra 包含测试、lint、构建、
元数据校验和绘图依赖。

## 检查 PyTorch 与设备

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.backends.mps.is_available()); print(torch.cuda.is_available())"
```

`device: auto` 依次选择 CUDA、Apple MPS 和 CPU。显式请求不可用加速器会失败。
学习和调试从 CPU 开始；只有流程正确后才改变硬件。

## 检查配置

```bash
uv run segment show-config --config configs/learning_minimal.yaml
```

确认 Python 能导入以 editable 方式安装的包，并且输出显示 CPU 有界子集、128 像素
图像和两轮训练。

## 有意识地排错

安装失败时，记录 Python 版本、操作系统、命令和第一个错误。理解原因前不要向托管
环境混入 pip 安装包。只有常规 CLI 错误细节不足时才使用 `--debug`。

下一章：[教程 02](02-data-and-masks.zh-CN.md)。

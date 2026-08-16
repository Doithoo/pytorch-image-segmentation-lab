# 包指南

[English](README.md) | [代码导览](../../docs/code-tour.zh-CN.md)

`image_segmenter` 是安装后的生产包。依赖从 CLI 编排指向聚焦的数据、模型、训练、
评估和推理模块。

## 从这里开始

先读 `config.py`，再读 `data/dataset.py`、`models/unet.py` 和
`training/losses.py`。它们定义系统其余部分传递的契约。追踪调用时运行 dry-run。

```bash
uv run segment train --config configs/learning_minimal.yaml --dry-run
```

## 模块职责

- `data`：校验、清单、变换、加载和原始标签映射。
- `models`：手写 U-Net、DeepLab 适配器和模型注册表。
- `training`：损失、循环、指标写入、checkpoint、恢复和设备。
- `evaluation`：指标表格、曲线和诊断可视化。
- `inference`：只依赖 checkpoint 的预测和输出保真。
- `cli.py`：参数解析和面向用户的错误边界。

## 修改规则

掩码保持类别索引，模型输出保持 logits，成对几何集中管理。Checkpoint 契约改变
需要 schema 决策以及迁移或明确拒绝。共享 trainer 行为不能按特定模型名分支。

## 公共接口

稳定的初学者接口是 `segment` CLI 和文档化模块类。下划线开头的 helper 属于内部
实现。在职责模块增加测试；跨多个模块的变化还要增加端到端测试。

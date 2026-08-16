# 测试指南

[English](README.md) | [贡献指南](../CONTRIBUTING.zh-CN.md)

测试使用微型合成 fixture，绝不能下载 Oxford、预训练权重或任何网络资源。

## 运行测试

```bash
uv run pytest
uv run pytest tests/test_end_to_end.py -v
```

红绿开发时运行聚焦文件，提交前运行完整套件。发布验证会把弃用警告当作错误。

## 测试层级

单元测试覆盖标签映射、成对变换、损失、指标、U-Net 形状、配置和 checkpoint
schema。集成测试覆盖清单、CLI、评估、推理、脚本和打包。端到端测试会准备合成配对、
在 CPU 训练、评估并预测。

## Fixtures

`paired_data` 创建对齐 RGB 图像和 Oxford 风格掩码。`oxford_data` 增加合成官方
划分文件。`trained_run` 创建微型真实 checkpoint。优先使用这些行为 fixture，
不要依赖 mock。

## 增加覆盖

先写一个聚焦的失败测试。断言可观察的文件、值、形状或错误，不要断言私有调用次数。
修复警告而不是过滤。把昂贵数据和模型下载留在 CI 之外。

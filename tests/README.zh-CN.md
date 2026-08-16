# 把测试当成可运行的行为说明

[English](README.md) | [贡献指南](../CONTRIBUTING.zh-CN.md)

使用项目前不必通读测试，但几组小输入比长文更适合理解边界情况。测试全部使用合成数据，
不下载 Oxford 或预训练权重。

## 初学者推荐先看

1. `test_metrics.py`：手算一个混淆矩阵，再对照 IoU/Dice。
2. `test_dataset.py`：看 1/2/3 怎样映射，以及未知值为何失败。
3. `test_examples.py`：五个教学示例应怎样运行。
4. `test_end_to_end.py`：从配对数据到训练、评测和预测的完整闭环。

```bash
uv run pytest tests/test_metrics.py -v
uv run pytest tests/test_end_to_end.py -v
uv run pytest
```

新增行为时先写一个会失败的聚焦测试，确认失败原因正确，再实现最小修改。断言用户能观察
的文件、值、形状和错误消息，不要锁定私有函数调用次数。

`conftest.py` 提供 `paired_data`、`oxford_data` 和 `trained_run` 等小型真实对象。优先
使用这些 fixture；网络、完整数据集和大模型不应进入测试。

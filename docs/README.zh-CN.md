# 文档导航

[English](README.md) | [项目 README](../README.zh-CN.md)

第一次运行项目，请使用项目 README 中最短的可执行流程。需要系统学习，或者只想查询
某一部分时，从本页进入。

## 从这里开始

- [实践路线](tutorial/learning-path.zh-CN.md)：从张量形状逐步走到一次受控模型对比。
- [教程索引](tutorial/README.zh-CN.md)：根据当前问题选择六章实操教程中的一章。

## 操作指南

这些文档对应在仓库中实际要完成的任务。

| 要做的事 | 文档 |
| --- | --- |
| 切换模型结构 | [使用模型](guides/using-models.zh-CN.md) |
| 准备另一份数据 | [使用自己的数据](guides/using-your-data.zh-CN.md) |
| 接入 Python 模型工厂 | [接入自己的模型](guides/adding-models.zh-CN.md) |
| 接入自定义 Dataset | [接入自己的 Dataset](guides/adding-datasets.zh-CN.md) |
| 公平比较多次运行 | [实验指南](guides/experiments.zh-CN.md) |
| 定位运行错误 | [排错指南](guides/troubleshooting.zh-CN.md) |

## 查询资料

需要确认支持的名称、字段或数据格式时使用这些文档。

- [模型列表](reference/model-zoo.zh-CN.md)
- [数据集格式](reference/dataset-formats.zh-CN.md)
- [配置字段](reference/config-reference.zh-CN.md)
- [VOC 2012 provider](reference/voc2012.zh-CN.md)

## 原理与代码

需要理解代码为什么这样组织时使用这些文档。

- [分割原理](concepts/how-it-works.zh-CN.md)
- [代码导览](concepts/code-tour.zh-CN.md)
- [配置流](concepts/configuration-flow.zh-CN.md)

## 实测记录

[U-Net 实测记录](recorded-run/README.zh-CN.md)保存一份真实运行的配置、环境、指标和
可视化证据。其他模型文档不提供未经验证的性能数字。

## 架构决策

[架构决策 0001](architecture/0001-extensible-models-and-datasets.zh-CN.md)说明模型、数据集、
标签模式、AMP 和运行对比为何采用当前扩展合同。

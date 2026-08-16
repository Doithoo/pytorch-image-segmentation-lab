# 核心代码包怎么读

[English](README.md) | [代码导览](../../docs/code-tour.zh-CN.md)

这里是安装后由 `segment` 命令调用的代码。第一次先运行 `examples/`，再把本页放在编辑器
旁边，沿调用顺序读正式实现。

```text
segment train
  -> cli.main()                    解析命令与显示错误
  -> load_config()                 合并并校验配置
  -> train_from_config()           创建数据、模型、loss 和 optimizer
  -> Trainer.fit()                 训练、验证和写指标
  -> save_checkpoint()             保存 best.pt / last.pt
```

| 目录/文件 | 输入 | 输出 | 什么时候读 |
| --- | --- | --- | --- |
| `config.py` | 默认值、YAML、覆盖 | 最终配置 | 参数不符合预期时 |
| `data/` | manifest、图片、原始 mask | Tensor batch | 学习数据管道时 |
| `models/` | 模型名与尺寸 | logits 模型 | 学 U-Net/换模型时 |
| `training/` | 配置、loader、模型 | checkpoint、metrics | 学优化循环时 |
| `evaluation/` | checkpoint、有标签 split | JSON、CSV、错误图 | 分析结果时 |
| `inference/` | checkpoint、新图片 | 原始/彩色/叠加 mask | 使用模型时 |
| `cli.py` | 终端参数 | 调用以上模块 | 理解应用入口时 |

第一遍推荐 `dataset.py -> transforms.py -> unet.py -> losses.py -> trainer.py`。配置严格解析、
manifest 审计、checkpoint RNG 和 CLI argparse 可以在遇到相关问题时再读。

修改代码时保持三个基本事实：mask 是整数类别，模型返回 logits，所有随机几何对 image
与 mask 使用同一选择。跨多个模块的行为变化需要端到端测试。

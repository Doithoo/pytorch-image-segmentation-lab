# 配置从哪里来，最后去了哪里

[English](configuration-flow.md) | [返回 README](../README.zh-CN.md)

项目只维护一份最终 `ExperimentConfig`。命令行不会在训练中途偷偷改变参数；所有来源
先合并、校验，再创建 Dataset 和模型。

## 覆盖顺序

```text
config.py 默认值
  < YAML 文件
  < --set KEY VALUE
  < --device / --output-dir 等专用参数
```

例如：

```bash
uv run segment show-config --config configs/learning_minimal.yaml \
  --set train.epochs 5 \
  --set loss.dice_weight 0.5 \
  --device cpu
```

`--set` 的值先按 JSON 解析，因此数字、布尔、null、列表和字典可以保持类型。字符串无法
解析为 JSON 时按普通文本处理。

## 为什么先运行 show-config

配置文件名只是输入之一。真正运行的值可能还受到默认值和 CLI 覆盖影响。

```bash
uv run segment show-config --config configs/learning_minimal.yaml
```

输出就是训练将使用的完整配置。训练开始后，同一份内容写入运行目录的 `config.yaml`。
比较实验时应比较这份解析结果，不要只比较两个 YAML 文件。

## 无效设置什么时候失败

配置加载阶段会拒绝未知键、错误类型、无效模型名、非正尺寸、错误标签映射、冲突的
ignore index、无效 loss 权重和不支持的设备。这样不会等到下载数据或训练十分钟后才
发现 `epohs` 拼错。

数据量是否足以生成非空 split 要在 `prepare-data` 扫描样本后才能判断，因此由 manifest
准备阶段报告。

## 配置怎样传入各模块

```text
CLI -> load_config/config_from_dict -> ExperimentConfig
  -> data.*      Dataset、transform、DataLoader
  -> model.*     U-Net/DeepLab 构造
  -> loss.*      CE/Dice 和 ignore index
  -> train.*     optimizer、scheduler、epoch
  -> top-level   device、输出目录、run name
```

较小的函数继续接收明确参数，例如 `map_mask(raw, label_map, ignore_values, ignore_index)`，
避免所有代码都依赖一个全局配置对象。

## checkpoint 与配置

checkpoint 保存训练时的完整配置。评测和推理优先使用 checkpoint 中的模型与预处理，
不要求用户再次提供 YAML。

训练恢复更严格：当前配置可以改变输出位置和无 scheduler 时的最终 epoch，但模型、数据
处理、loss、batch、标签与 manifest 必须一致。cosine 的总 epoch 属于调度计划，不能在
恢复时修改。

完整字段说明见[配置参考](config-reference.zh-CN.md)。

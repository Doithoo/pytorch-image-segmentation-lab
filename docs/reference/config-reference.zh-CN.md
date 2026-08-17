# 配置参数参考

[English](config-reference.md) | [文档导航](../README.zh-CN.md)

先用现成配置跑通，再回来查字段。精确默认值以 `segment show-config` 和
`src/image_segmenter/config.py` 为准；这里解释每个字段何时需要修改。

## data

| 字段 | 含义 | 初学者建议 |
| --- | --- | --- |
| `data_dir` | 原始数据根目录 | Oxford 使用 `data/raw` |
| `manifest_dir` | train/valid/test CSV 目录 | 每个新数据集使用独立目录 |
| `image_size` | 网络输入尺寸 | U-Net 每个维度需为 16 的倍数；先查看模型限制 |
| `resize_mode` | `stretch` 或 `letterbox` | 长宽比敏感的数据优先尝试 `letterbox` |
| `crop_size` | 训练时的成对裁剪尺寸 | 必须不大于 `image_size` |
| `normalize_mean/std` | RGB 归一化 | 与训练和推理保持一致 |
| `label_map` | 原始像素值到从 0 开始的连续类别 ID | 先打印原始 mask 唯一值 |
| `ignore_values` | 不参与 loss/指标的原始值 | Oxford 为 `[3]` |
| `split_ratios` | 通用数据 train/valid/test 比例 | 三项和为 1，且实际样本不能产生空 split |
| `seed` | 通用切分或 Oxford trainval 切分种子 | 比较实验时固定 |
| `num_workers` | DataLoader 子进程数 | 先用 0，稳定后再增加 |
| `pin_memory` | CUDA 主机页锁定内存 | CPU/MPS 通常保持 false |
| `horizontal_flip` | 训练时水平翻转概率 | 先用 0.5；方向有语义时关闭 |
| `max_*_samples` | 各 split 的样本上限 | 入门配置用于缩短运行；正式结果设 null |

图片和掩码都会缩放为 `image_size x image_size`；图片双线性、掩码最近邻。当前配置没有
padding/crop 模式。

## model

| 字段 | 含义 | 限制/建议 |
| --- | --- | --- |
| `name` | `list-models` 中的注册模型名 | 第一次用 U-Net；模型列表见[模型列表](model-zoo.zh-CN.md) |
| `in_channels` | 输入通道数 | 当前 RGB 流程必须为 3 |
| `expected_num_classes` | 与 `dataset.yaml` 标签模式核对的类别数 | 多分类时按元数据填写 |
| `params.base_channels` | U-Net 起始宽度 | 8/16 适合练习，32 用于参考运行 |
| `pretrained` | 是否加载预训练权重 | U-Net 必须 false；DeepLab true 会联网 |

`base_channels` 也传入统一模型工厂，但 DeepLab 不使用它。预训练与随机初始化对比时要
把初始化差异写进结论。

## loss

| 字段 | 含义 | 示例 |
| --- | --- | --- |
| `name` | 可读的 loss 方案名称 | `cross_entropy` / `cross_entropy_dice` |
| `cross_entropy_weight` | CE 系数，非负 | 入门为 1.0 |
| `dice_weight` | foreground soft Dice 系数，非负 | 参考为 0.5 |
| `ignore_index` | 映射后忽略值 | Oxford 为 255，不能是 0/1 |

两个权重不能同时为 0。`name` 用于说明意图，实际组合由权重决定。

## train

| 字段 | 含义 | 初学者建议 |
| --- | --- | --- |
| `epochs` | 计划总轮数 | 先 2；参考运行 25 |
| `batch_size` | 每次优化的图片数 | OOM 时先减半 |
| `lr` | AdamW 学习率 | 从现成配置开始，不要盲目照搬其他任务 |
| `weight_decay` | AdamW 权重衰减 | 基线保持 `1e-4` |
| `optimizer` | 优化器 | 当前只支持 `adamw` |
| `scheduler` | `none` 或 `cosine` | 短练习 none，长运行 cosine |
| `grad_clip` | 梯度范数上限，0 表示关闭 | 出现不稳定前保持 0 |
| `seed` | 模型、shuffle 和变换 seed | 受控对比时固定 |

cosine 的 `T_max` 使用计划总 epoch。恢复 cosine checkpoint 时不能修改 `epochs`；无
scheduler 的运行可以增加总轮数继续训练。

## 顶层字段

| 字段 | 含义 |
| --- | --- |
| `device` | `auto`、`cpu`、`cuda` 或 `mps` |
| `output_dir` | 所有本地运行产物的根目录 |
| `run_name` | 子目录名；null 时自动生成 |
| `log_level` | 可填 `debug/info/warning/error/critical`；当前只解析和校验，尚不改变 CLI 输出 |

每个实验使用独立 `run_name`。已含 checkpoint 的目录不会被新训练覆盖；恢复时 checkpoint
必须属于目标运行目录。`log_level` 是为配置兼容保留的字段，不应把它当作当前可用的
日志开关。

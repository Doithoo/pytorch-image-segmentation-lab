# ADR 0001：可扩展模型与数据集架构

[English](0001-extensible-models-and-datasets.md)

- 状态：已接受
- 日期：2026-08-16
- 范围：2D 单标签语义分割

## 背景

当前项目提供手写 `unet_small` 和 torchvision `deeplabv3_resnet50`，配置与数据流程
主要围绕 Oxford-IIIT Pet 二分类任务设计。学习者可以完成一次训练，但很难自然过渡到
模型对比、多分类掩码、自定义数据协议或仓库外模型。

本次升级的目标不是堆积模型名称，而是建立三种可迁移能力：通过配置切换内置模型、
注册或导入自己的模型、把训练流程用于自己的多分类数据。现有 Oxford 参考运行和二分类
checkpoint 必须继续可解释和可使用。

## 决策

采用分层注册与 provider 架构：

```text
YAML / CLI
   |
   +-> ModelRegistry -> ModelFactory -> OutputAdapter
   |
   +-> DatasetRegistry -> DatasetFactory -> LabelSchema
   |
   `-> PreflightValidator -> Trainer / Evaluator / Predictor
```

模型与数据各提供内置注册入口和显式 Python 工厂入口。第三方模型库作为可选依赖，不
影响核心安装。框架不做自动插件发现，也不把可执行代码写进 checkpoint。

## 模型系统

核心安装维护以下模型：

| 注册名 | 来源 | 主要比较点 |
| --- | --- | --- |
| `unet_small` | 本项目 | 完整可读的 encoder、skip 与 decoder |
| `fcn_resnet50` | torchvision | 经典 FCN 基线 |
| `deeplabv3_resnet50` | torchvision | 空洞卷积与 ASPP |
| `deeplabv3_mobilenet_v3_large` | torchvision | 轻量 backbone 下的 DeepLabV3 |
| `lraspp_mobilenet_v3_large` | torchvision | 低计算量分割结构 |

`model-zoo` 可选依赖安装 `segmentation-models-pytorch`，注册 `smp_unet`、
`smp_unetplusplus`、`smp_fpn` 和 `smp_pan`。encoder 及 decoder 参数进入
`model.params`，不为每种组合创建新注册名。

自定义模型可以在仓库内调用 `register_model()`，也可以在配置中填写工厂路径：

```yaml
model:
  name: my_segmenter
  factory: my_project.models:create_model
  pretrained: false
  params:
    width: 32
```

工厂签名为：

```python
def create_model(
    *,
    in_channels: int,
    num_classes: int,
    pretrained: bool,
    **params,
) -> torch.nn.Module: ...
```

模型接收 `[B,C,H,W]` 浮点 tensor，返回 `[B,K,H,W]` logits。模型内部不执行
softmax 或 argmax。torchvision 的 `{"out": logits}` 由注册项指定的 adapter 转换；
直接 tensor 不再包装。输出空间尺寸不一致时预检查报错，不静默插值。

`ModelRegistry` 保存来源、工厂、依赖、参数说明、预训练支持、输入约束和输出 adapter。
checkpoint 保存注册名、provider、解析后的参数和权重来源。外部工厂无法导入时，恢复
操作显示原始模块和函数路径，不回退到另一个模型。

## 数据与标签契约

每份准备完成的数据在 manifest 目录保存 `dataset.yaml`：

```yaml
schema_version: 1
name: my-road-scenes
classes:
  - {id: 0, name: background, color: [0, 0, 0]}
  - {id: 1, name: road, color: [128, 64, 128]}
  - {id: 2, name: vehicle, color: [0, 0, 142]}
ignore_index: 255
source:
  provider: paired_masks
  split_method: grouped
  seed: 666
counts: {train: 420, valid: 90, test: 90}
```

`LabelSchema` 由类别 ID、名称、显示颜色和 ignore index 组成。类别 ID 必须从 0 连续
排列到 K-1，名称与颜色不得重复，ignore index 不能占用类别 ID。训练从准备完成的数据
读取 schema；配置描述原始数据怎样映射到它。checkpoint 保存 schema 与数据清单哈希。
模型类别数由 schema 推导，不由模型配置独立决定。

内置数据 provider：

- `oxford_pet`：保留官方 test，并在 breed 内划分 train/valid。
- `voc2012`：支持 21 类和 ignore 255；真实数据为可选示例，不进入默认下载或 CI。
- `paired_masks`：递归匹配同名图片与掩码，再确定性切分。
- `split_folders`：读取已有的 train/valid/test 目录。
- `manifest`：验证显式图片、掩码、split 和可选 `group_id`。

核心读取器支持 8 位或 16 位整数 mask、PNG palette 索引和显式颜色映射的 RGB mask。
多个原始值可以映射到同一训练类别。概率图、one-hot mask、多标签像素、实例 ID mask
以及未声明的值或颜色会被拒绝。

外部数据工厂按 split 返回 Dataset，调用签名为：

```python
def create_dataset(
    *,
    split: str,
    data_dir: pathlib.Path,
    manifest_dir: pathlib.Path,
    transform: collections.abc.Callable,
    label_schema: LabelSchema,
    **params,
) -> torch.utils.data.Dataset: ...
```

每条样本返回：

```python
image: float Tensor[C, H, W]
target: long Tensor[H, W]
sample_id: str
```

框架向工厂传入成对 transform 和 `LabelSchema`。患者、视频、地理区域等相关样本使用
显式 split、`group_id` 或外部工厂，不能默认按单张图片随机拆分。

预处理支持 `stretch` 与保持长宽比的 `letterbox`；后者使用 ignore index 填充 mask。
训练可启用成对随机裁剪和水平翻转。`image_size` 接受整数或 `[height, width]`。

## 配置与兼容

新配置使用 `config_version: 2`。无版本的现有配置按 v1 读取并迁移到内部结构：

- `model.base_channels` 迁移到 `model.params.base_channels`。
- `model.num_classes` 作为旧配置的预期值，与 `LabelSchema` 核对。
- 原有 `data.label_map` 和 `data.ignore_values` 继续接受。
- `show-config` 输出解析后的 v2 配置，使迁移结果可见。

`unet_small`、`deeplabv3_resnet50`、现有 Oxford 配置和二分类 checkpoint 保持兼容。
Oxford 继续以宠物类别 IoU 选择最佳 checkpoint；通用多分类配置默认使用 mean IoU。
`train.best_metric` 支持 `mean_iou`、`mean_dice`、`pixel_accuracy` 和
`class_iou:<class-name>`。

## CLI 与错误处理

新增发现和检查命令：

```bash
uv run segment list-models
uv run segment model-info lraspp_mobilenet_v3_large
uv run segment list-datasets
uv run segment dataset-info paired_masks
uv run segment inspect-data --config configs/my_data.yaml --limit 6
uv run segment compare-runs artifacts/compare-* --metric mean_iou
```

缺少可选依赖时，模型仍出现在列表中并显示
`uv sync --extra model-zoo`。未知名称提供相近注册名；provider 参数错误显示字段、收到
的值和允许范围；外部工厂错误区分模块、函数、签名和返回契约。类别或输出不匹配时同时
显示 schema、target 唯一值和 logits shape。

训练在创建正式运行前完成一次预检查：解析 provider、读取 schema、核对 manifest、
构造模型和一个 batch，再验证输入、标签与输出契约。

`compare-runs` 只汇总配置差异、最佳指标、参数量和运行时间，不替用户判断模型优劣。

## CUDA、MPS 与 Kaggle

`train.amp` 控制自动混合精度。AMP 仅在 CUDA 启用，并保存 GradScaler 状态以支持恢复。
旧 checkpoint 没有 scaler 状态时按未启用 AMP 恢复；不能在同一次恢复中改变 AMP
语义。CPU 或 MPS 请求 AMP 时给出明确说明并使用全精度。此次不加入 DDP、
`torch.compile` 或多 GPU 训练。

环境文档分别说明 CPU、NVIDIA CUDA、Apple MPS 和 Kaggle GPU：

- CUDA 指南检查驱动、PyTorch 构建、设备名称、显存和 OOM 调整顺序。
- MPS 指南说明统一内存、支持边界和回退方式。
- Kaggle 指南覆盖启用 GPU、安装、只读 `/kaggle/input`、可写
  `/kaggle/working`、数据准备、dry run、训练恢复和产物保存。
- 不写固定训练时长、免费额度或未经实际运行验证的设备结论。
- Kaggle 提供多张 GPU 时仍使用单卡，因为当前训练器不支持分布式训练。

## 文档与示例

README 只展示能力入口和代表模型。详细内容拆分为模型列表、模型使用、自定义模型、数据
格式、自定义数据、外部数据工厂和 VOC 示例。教程 02 到 05 分别扩展标签 schema、模型
选择、模型相关训练设置和多分类评测。

可运行扩展示例位于 `examples/extensions/`，包含外部模型和数据工厂；配套配置直接引用
这些路径，并由测试导入执行。

文档使用事实化建议，不替用户设定进度、成就或能力结论。建议应附适用条件和限制。
中文按自然中文组织，不要求与英文拥有完全相同的章节结构。代码注释只解释动态导入、
注册副作用、输出适配和标签映射等非显然逻辑。

## 验证边界

本次不为每个新增模型执行完整训练，也不发布未经完整实验得到的 mIoU。现有 U-Net 是
唯一保留实测结果的完整参考运行；其他模型称为可切换模型或示例配置。

自动验证包括：

- 所有核心模型在无预训练权重时通过输出契约测试。
- 独立 CI job 安装 `model-zoo` 并离线测试 SMP 模型。
- 合成三分类数据完成准备、短训练、评测和推理。
- 覆盖整数、palette、RGB、16 位 mask 与 group 泄漏检查。
- 使用合成目录测试 VOC provider，不在 CI 下载真实 VOC。
- 覆盖外部工厂成功、导入失败、签名错误和返回契约错误。
- 覆盖 v1 配置、二分类 checkpoint、cosine 恢复与随机状态兼容。
- 覆盖 AMP 配置、非 CUDA 行为和 scaler checkpoint；普通 CI 不声称验证 CUDA kernel。
- 验证示例配置、文档命令、模型名、依赖名、工厂路径和相对链接。

Mac 本地只运行分钟级合成数据端到端测试、模型 forward、dry run、静态检查和打包，不
需要数小时的完整训练。CUDA/Kaggle 文档提供 smoke 命令，但不会把未实际执行的结果写成
已验证事实。

## 不包含的范围

- 实例分割、全景分割、检测、3D 分割和多标签像素。
- 自动发现第三方插件或序列化用户代码。
- 多 GPU、分布式训练、模型性能排行榜和所有模型的预训练 checkpoint。
- 为每种模型与 encoder 组合复制配置文件。

## 结果

该设计增加一定的注册与迁移代码，但把模型、数据和训练之间的隐式耦合转为可验证契约。
核心安装保持轻量；进阶模型只在安装可选依赖后启用。新增 provider 需要同时维护注册
元数据、契约测试和用户文档。

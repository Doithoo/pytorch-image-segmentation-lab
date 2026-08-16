# 配置参考

[English](config-reference.md) | [返回 README](../README.zh-CN.md)

所有 YAML 键都映射到 `config.py` 中的类型化 dataclass。本页解释意图；精确的当前
默认值请使用 `show-config` 查看。

## 顶层字段

- `device`：`auto`、`cpu`、`cuda` 或 `mps`。Auto 依次选择 CUDA、MPS、CPU；
  显式指定不可用设备会报错。
- `output_dir`：运行产物根目录。
- `run_name`：稳定的子目录名；省略时自动生成。

## 数据字段

- `data.manifest_dir`：包含 `train.csv`、`valid.csv` 和 `test.csv` 的目录。
- `data.image_size`：方形网络尺寸，必须为正且能被 16 整除。
- `data.label_map`：原始整数到类别索引的映射。Oxford 使用 `{1: 1, 2: 0}`。
- `data.ignore_values`：映射到损失忽略索引的原始值；Oxford 使用 `[3]`。
- `data.normalize_mean/std`：三个图像通道值。
- `data.horizontal_flip`：训练专用的 0 到 1 概率。
- `data.num_workers`、`pin_memory`：DataLoader 控制项。
- `data.max_*_samples`：入门实验可选的有界子集。

## 模型与损失字段

- `model.name`：`unet` 或 `deeplabv3_resnet50`。
- `model.in_channels`：RGB 使用 3。
- `model.num_classes`：背景/宠物使用 2。
- `model.base_channels`：U-Net 第一层宽度；越大越占内存。
- `model.pretrained`：进阶 DeepLab 权重开关；可能下载权重。
- `loss.cross_entropy_weight`、`loss.dice_weight`：非负，且总和必须为正。
- `loss.ignore_index`：映射后目标值 255。

## 训练字段

- `train.batch_size`、`train.epochs`、`train.lr`、`train.weight_decay`：
  常规优化控制项。
- `train.optimizer`：当前为 `adamw`。
- `train.scheduler`：`none` 或 `cosine`。
- `train.seed`：设置 Python、NumPy、PyTorch、worker 和采样随机种子。
- `train.gradient_clip_norm`：可选最大梯度范数。

## 校验配置文件

```bash
uv run segment show-config --config configs/reference_unet.yaml
```

带注释的文件说明见 [configs](../configs/README.zh-CN.md)。图像尺寸、归一化、标签
或模型构造的变化会影响 checkpoint 兼容性，应创建新的运行。

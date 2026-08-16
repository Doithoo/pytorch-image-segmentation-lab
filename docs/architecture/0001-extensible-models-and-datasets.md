# ADR 0001: Extensible Models and Datasets

[简体中文](0001-extensible-models-and-datasets.zh-CN.md)

- Status: Accepted
- Date: 2026-08-16
- Scope: 2D single-label semantic segmentation

## Context

The project currently exposes a handwritten `unet_small` and torchvision
`deeplabv3_resnet50`. Its configuration and data flow are centered on binary
Oxford-IIIT Pet segmentation. A learner can finish one run, but the path to
model comparison, multiclass masks, custom data protocols, and external models
is not yet explicit.

The goal is not a long model menu. The project should teach three transferable
operations: switch maintained models through configuration, register or import
a custom model, and apply the same training system to custom multiclass data.
Existing Oxford results and binary checkpoints must remain usable.

## Decision

Adopt layered registries and providers:

```text
YAML / CLI
   |
   +-> ModelRegistry -> ModelFactory -> OutputAdapter
   |
   +-> DatasetRegistry -> DatasetFactory -> LabelSchema
   |
   `-> PreflightValidator -> Trainer / Evaluator / Predictor
```

Models and datasets support maintained registrations and explicit Python factory
paths. Third-party model libraries remain optional. The framework will not use
automatic plugin discovery or serialize executable user code in checkpoints.

## Model System

The core installation maintains `unet_small`, `fcn_resnet50`,
`deeplabv3_resnet50`, `deeplabv3_mobilenet_v3_large`, and
`lraspp_mobilenet_v3_large`. These represent a readable encoder-decoder,
classic FCN, ASPP with two backbones, and a lower-compute architecture.

The optional `model-zoo` extra installs `segmentation-models-pytorch` and
registers `smp_unet`, `smp_unetplusplus`, `smp_fpn`, and `smp_pan`. Encoder and
decoder choices belong in `model.params`; model-encoder combinations do not get
separate registry names.

Repository models use `register_model()`. External models use a configured
`module:function` factory with this contract:

```python
def create_model(
    *,
    in_channels: int,
    num_classes: int,
    pretrained: bool,
    **params,
) -> torch.nn.Module:
    ...
```

Models accept float `[B,C,H,W]` tensors and return `[B,K,H,W]` logits without
softmax or argmax. A registered adapter extracts torchvision `{"out": logits}`
outputs. Spatial mismatches fail preflight instead of being silently resized.

Registry metadata includes provider, factory, dependency, parameter guidance,
pretraining support, input constraints, and output adapter. Checkpoints record
the resolved name, provider, parameters, and weight source. Missing external
factories produce an actionable error and never fall back to another model.

## Data and Label Contract

Prepared manifest directories contain a versioned `dataset.yaml` with class
IDs, names, display colors, ignore index, provider, split method, seed, and
counts. Class IDs are contiguous from zero through K-1; names and colors are
unique; the ignore index is not a class ID. `LabelSchema` is the source of class
count and label semantics. Training reads it from prepared data, while
configuration describes how raw source values map into it. Checkpoints retain
the schema and manifest hashes.

Maintained providers are:

- `oxford_pet`, preserving the official test set and breed-aware train/valid split;
- `voc2012`, supporting 21 classes and ignore 255 as an optional real dataset;
- `paired_masks`, matching recursive same-stem image and mask paths;
- `split_folders`, reading existing train/valid/test directories;
- `manifest`, validating explicit paths, splits, and optional `group_id`.

Core readers support 8-bit and 16-bit integer masks, PNG palette indices, and
RGB masks with an explicit color map. Several raw values may map to one training
class. Probability maps, one-hot masks, multilabel pixels, instance IDs, and
undeclared values or colors fail validation.

External factories return one Dataset per split with this contract:

```python
def create_dataset(
    *,
    split: str,
    data_dir: pathlib.Path,
    manifest_dir: pathlib.Path,
    transform: collections.abc.Callable,
    label_schema: LabelSchema,
    **params,
) -> torch.utils.data.Dataset:
    ...
```

Each item is a float `Tensor[C,H,W]`, long `Tensor[H,W]`, and string sample ID.
The framework passes paired transforms and `LabelSchema`. Related patients,
video frames, or regions must use explicit splits, `group_id`, or an external
factory.

Preprocessing supports `stretch` and aspect-preserving `letterbox`; letterbox
mask padding uses the ignore index. Training may apply paired random crops and
horizontal flips. `image_size` accepts an integer or `[height, width]`.

## Configuration and Compatibility

New files use `config_version: 2`. Unversioned existing files load as v1 and are
migrated internally. `model.base_channels` moves to
`model.params.base_channels`; `model.num_classes` becomes a legacy expectation
checked against `LabelSchema`; existing label and ignore maps remain accepted.
`show-config` displays the resolved v2 form.

Existing model names, Oxford configurations, and binary checkpoints remain
supported. Oxford selects checkpoints with pet-class IoU. General multiclass
runs default to mean IoU. `train.best_metric` accepts `mean_iou`, `mean_dice`,
`pixel_accuracy`, and `class_iou:<class-name>`.

## CLI and Failures

Discovery and inspection commands include `list-models`, `model-info`,
`list-datasets`, `dataset-info`, `inspect-data`, and `compare-runs`. Missing
optional dependencies remain visible and include the exact install command.
Unknown names suggest close matches. Provider errors distinguish invalid
parameters, import failures, function lookup, signatures, and return contracts.
Label/output mismatches report the schema, target values, and logits shape.

Before creating a full run, preflight resolves providers, reads the schema,
checks manifests, builds one batch and a model, and validates their contracts.
`compare-runs` summarizes configuration differences, best metrics, parameter
counts, and runtime without declaring a universally best model.

## CUDA, MPS, and Kaggle

`train.amp` enables CUDA automatic mixed precision and saves GradScaler state
for resume. A legacy checkpoint without scaler state resumes as a non-AMP run;
AMP semantics cannot change during one resumed run. CPU or MPS requests receive
an explicit notice and use full precision. DDP, `torch.compile`, and multi-GPU
training are outside this change.

Environment documentation covers CPU, NVIDIA CUDA, Apple MPS, and Kaggle GPU.
The Kaggle path includes accelerator setup, installation, read-only
`/kaggle/input`, writable `/kaggle/working`, data preparation, dry run, resume,
and artifact retention. Documentation avoids fixed runtime, quota, and
unverified hardware claims. A multi-GPU Kaggle session still uses one GPU.

## Documentation and Examples

The README links to focused guides for model discovery, model comparison,
custom models, mask formats, custom datasets, external data factories, and VOC.
Tutorials 02 through 05 expand to label schemas, architecture choices,
model-dependent training settings, and multiclass evaluation.

Runnable external model and dataset examples live under
`examples/extensions/`, with configurations that import them and tests that
execute them.

Documentation gives conditional, factual advice. It avoids gamified completion
language, mandatory progress, and claims that one command confers mastery.
Chinese documents use natural Chinese structure rather than mirroring English
headings. Comments are reserved for dynamic imports, registration side effects,
output adaptation, and label mapping that code alone does not make clear.

## Verification Boundary

This change does not fully train every new model or publish unmeasured mIoU.
The measured U-Net run remains the only complete reference result; other models
are switchable implementations and example configurations.

Automated verification covers core and optional model contracts, a synthetic
three-class prepare/train/evaluate/predict flow, mask encodings, grouped splits,
synthetic VOC layout, external factory failures, v1 configuration and binary
checkpoint compatibility, AMP state handling, example configurations, CLI
names, documentation links, typing, packaging, and formatting.

Local Mac verification uses short synthetic runs, forwards, dry runs, static
checks, and packaging. It does not require multi-hour training. Ordinary CI
cannot prove CUDA kernel behavior; CUDA and Kaggle guides provide smoke commands
without presenting unexecuted results as verified.

## Out of Scope

- Instance, panoptic, detection, 3D, and multilabel segmentation.
- Automatic third-party discovery or serialization of user code.
- Distributed and multi-GPU training, leaderboards, and pretrained checkpoints
  for every model.
- One configuration file for every model and encoder combination.

## Consequences

The design adds registry and migration code but turns hidden model-data coupling
into explicit, testable contracts. The core install stays small; advanced model
families require an optional extra. Every new provider must maintain registry
metadata, contract tests, and user documentation.

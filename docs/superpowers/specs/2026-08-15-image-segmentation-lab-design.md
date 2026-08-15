# PyTorch Image Segmentation Lab Design

Date: 2026-08-15
Status: Approved
Repository: `Doithoo/pytorch-image-segmentation-lab`

## 1. Purpose

Build an open source, beginner-oriented semantic segmentation project that is a
sibling to `pytorch-image-classification-lab`. A learner who understands basic
Python, tensors, loss, and gradients should be able to complete a reproducible
image segmentation workflow without already knowing segmentation architectures
or research papers.

The project teaches one complete workflow:

```text
inspect image/mask pairs
-> validate and split data
-> run one batch
-> train a small U-Net
-> evaluate IoU and Dice
-> inspect overlays and worst cases
-> compare a pretrained DeepLabV3 model
```

The expected guided learning time is 8-12 hours. The first real run must work on
a CPU. Full reference experiments may use CUDA or Apple MPS.

## 2. Success Criteria

The first release is successful when all of the following are true:

- A fresh Python environment can install, test, and build the package.
- A CPU user can run a dry run and a two-epoch learning workflow.
- The data, model, loss, evaluation, and inference paths agree on label and
  `ignore_index` semantics.
- The handwritten U-Net and torchvision DeepLabV3 use the same training,
  evaluation, checkpoint, and prediction interfaces.
- Evaluation produces numeric evidence and inspectable visual outputs.
- A published U-Net reference run reaches at least 0.70 pet foreground IoU on
  the untouched official test set. Model and configuration selection use the
  validation set, not the test set.
- Every published reference result includes its effective configuration,
  environment, manifest checksum, checkpoint, curves, metrics, and example
  predictions.
- The English and Simplified Chinese core documentation describe the same
  behavior and commands.

The primary goal is a clear and reproducible learning experience, not a
state-of-the-art leaderboard result.

## 3. Scope

### 3.1 Included

- Binary semantic segmentation using Oxford-IIIT Pet.
- A handwritten small U-Net as the teaching model.
- A pretrained torchvision DeepLabV3-ResNet50 comparison.
- Strict YAML configuration with command-line overrides.
- Dataset download, validation, deterministic manifests, and preview tools.
- Paired image/mask transforms.
- Cross-entropy and foreground soft Dice loss.
- Training, best/last checkpoints, and checkpoint resume.
- IoU, Dice, pixel accuracy, per-image metrics, overlays, and worst cases.
- Single-image and directory prediction.
- Paired-directory custom datasets.
- English and Simplified Chinese learning material.
- Unit, integration, CLI, packaging, and CPU end-to-end tests.

### 3.2 Not Included In The First Release

- Instance or panoptic segmentation.
- Object detection, bounding boxes, or Mask R-CNN.
- COCO, Pascal VOC, or other annotation format parsers.
- A general plugin framework for arbitrary datasets and trainers.
- A model zoo beyond U-Net and DeepLabV3-ResNet50.
- Albumentations or `segmentation-models-pytorch`.
- Notebooks as a required learning or execution path.
- AMP, distributed training, test-time augmentation, ONNX, or Gradio.
- A shared package dependency on `pytorch-image-classification-lab`.

These items can be considered in later, independently designed releases.

## 4. Repository Identity And Layout

The repository name is `pytorch-image-segmentation-lab`, the Python package is
`image_segmenter`, and the installed command is `segment`. The project uses the
MIT license and a `src` package layout.

```text
pytorch-image-segmentation-lab/
|-- configs/
|-- data/
|   |-- manifests/
|   `-- raw/
|-- docs/
|   |-- assets/
|   |-- superpowers/
|   |-- tutorial/
|   `-- *.md
|-- examples/
|-- scripts/
|-- src/image_segmenter/
|   |-- data/
|   |-- evaluation/
|   |-- inference/
|   |-- models/
|   |-- training/
|   |-- utils/
|   |-- cli.py
|   |-- config.py
|   `-- py.typed
|-- tests/
|-- CONTRIBUTING.md
|-- CONTRIBUTING.zh-CN.md
|-- LICENSE
|-- Makefile
|-- README.md
|-- README.zh-CN.md
`-- pyproject.toml
```

The two lab repositories should feel related through directory names,
configuration precedence, CLI conventions, artifact naming, bilingual docs,
and testing style. They do not share runtime code. Segmentation-specific
concepts must remain explicit instead of being forced through classification
abstractions.

## 5. Architecture And Contracts

The main tensor contract is:

```text
Dataset
  -> image: float32 [3, H, W]
  -> mask: int64 [H, W]
Model
  -> logits: float32 [B, 2, H, W]
Loss
  -> scalar computed only from mask != 255
Prediction
  -> class indices: int64 [B, H, W]
Metrics
  -> confusion counts and scores from valid pixels only
```

Module responsibilities are intentionally narrow:

- `data` pairs files, validates samples, creates manifests, maps labels, and
  returns transformed tensors.
- `models` converts image tensors into same-resolution two-class logits and has
  no knowledge of manifests or artifact directories.
- `training` assembles the model, data loaders, loss, optimizer, scheduler, and
  checkpoint lifecycle. Metric formulas live outside the trainer.
- `evaluation` computes metrics and creates tables and visual evidence without
  modifying model state.
- `inference` restores a self-contained checkpoint and predicts images without
  requiring the original training YAML.
- `cli` handles arguments and user-facing errors; it does not implement domain
  logic.

Small examples may duplicate a few lines from the package when that makes the
learning concept visible. They must identify what the full implementation adds
and link to it.

## 6. Learning Path And Commands

The first-run workflow is:

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

python scripts/download_data.py
segment prepare-data --set data.data_dir data/raw
python scripts/preview_dataset.py --data-dir data/raw

segment train --config configs/learning_minimal.yaml --dry-run
segment train --config configs/learning_minimal.yaml \
  --set train.epochs 2 --set run_name first-run

segment evaluate --checkpoint artifacts/first-run/best.pt --plot
segment predict --checkpoint artifacts/first-run/best.pt \
  --image path/to/image.jpg --output prediction.png
```

The examples are read and run in this order:

```text
examples/01_mask_tensor.py
examples/02_paired_transforms.py
examples/03_dataset_batch.py
examples/04_minimal_training.py
examples/05_predict_overlay.py
```

`configs/learning_minimal.yaml` uses 256 training, 64 validation, and 64 test
samples selected deterministically from the full manifests. It uses 128 x 128
inputs, a U-Net with 16 base channels, a batch size of 8, and two epochs. The
subset limits are explicit configuration values and can be removed without
regenerating manifests.

`configs/reference_unet.yaml` uses the full manifests, 256 x 256 inputs, a U-Net
with 32 base channels, a batch size of 8, and 25 epochs. It uses AdamW with a
learning rate of `3e-4`, weight decay of `1e-4`, and cosine decay.

`configs/deeplabv3_resnet50.yaml` uses the full manifests, 256 x 256 inputs, a
batch size of 4, and 15 epochs. It uses AdamW with a learning rate of `1e-4`,
weight decay of `1e-4`, and cosine decay. A measured run may motivate a later
configuration change, but the test set must not be used to choose it.

## 7. Dataset And Manifests

### 7.1 Oxford-IIIT Pet

The download script fetches the official image and annotation archives from
pinned official URLs, verifies recorded checksums, and only then moves them into
the final data directory. Interrupted or invalid downloads cannot appear as
complete datasets. The documentation includes the upstream license and
attribution.

The official `trainval` set is split deterministically into train and validation
sets within each of the 37 breeds, using 80 percent for training and 20 percent
for validation. The official test set remains untouched. The split seed is
stored in metadata.

Each manifest row contains at least:

```text
sample_id,image_path,mask_path,split,width,height
```

Oxford manifests also record breed and upstream split. `source.yaml` and
`summary.txt` record source information, label mapping, split seed, counts,
exact-file duplicate checks, and manifest checksums.

### 7.2 Label Semantics

The Oxford trimap mapping is fixed:

```text
raw 1 -> pet class 1
raw 2 -> background class 0
raw 3 -> ignore value 255
```

The official `3` label means `not classified` and commonly covers uncertain
boundary pixels. It is not trained as a semantic boundary class. The loss,
metrics, previews, overlays, and saved metadata all use the same mapping.

An advanced experiment may compare this default with treating all raw labels as
trainable, but it is not part of the primary workflow.

### 7.3 Validation

Data preparation validates all samples before writing usable manifests:

- every image has exactly one mask and every mask has one image;
- files decode successfully;
- image and mask dimensions match;
- raw masks contain only configured pixel values;
- each sample has at least one valid pixel;
- sample identifiers are unique;
- exact image or mask content is not duplicated across splits.

Empty-foreground samples are reported as warnings and retained because they can
be valid negative examples in custom datasets. Fatal problems are collected in
`data/manifests/invalid.csv`, summarized once, and cause the command to fail.

### 7.4 Custom Paired Directories

The first release supports this input convention:

```text
dataset/
|-- images/
|   |-- sample_001.jpg
|   `-- sample_002.jpg
`-- masks/
    |-- sample_001.png
    `-- sample_002.png
```

Files pair by relative path stem, so nested subdirectories are allowed but stems
must be unique within each relative directory. The user explicitly supplies the
raw-value-to-class mapping and `ignore` values. The project never infers mask
meaning from colors. The same validation and deterministic manifest process is
used for Oxford and custom data.

If the user supplies only one unsplit paired dataset, the default split is
80/10/10 with a fixed seed. Explicit train, validation, and test directories
take precedence and are never reshuffled.

## 8. Paired Transforms

All geometric randomness is sampled once and applied to the image and mask
together. Image resize uses bilinear interpolation; mask resize always uses
nearest-neighbor interpolation. Photometric transforms affect only the image.

The minimal training transform contains resize, optional horizontal flip,
tensor conversion, and normalization. Validation and inference transforms are
deterministic. The checkpoint stores input size, normalization, and class
mapping.

The preview script displays the original image, raw mask, mapped mask, and
overlay. It can render a deterministic sample grid and must visibly distinguish
ignored pixels from foreground, background, and prediction errors.

## 9. Models

### 9.1 Small U-Net

The teaching U-Net is implemented locally. With the default base width it uses
encoder widths 16, 32, 64, and 128, followed by a 256-channel bottleneck and a
mirrored decoder. Each stage uses two convolution, normalization, and ReLU
operations. The decoder upsamples, concatenates the matching encoder feature,
and applies its convolution block. A final 1 x 1 convolution produces two
logit channels.

The model supports a configurable base width. Input height and width must be
divisible by 16; configuration validation reports this before training.

### 9.2 DeepLabV3-ResNet50

The comparison model is provided by torchvision. When `pretrained: true`, it
loads `DeepLabV3_ResNet50_Weights.DEFAULT` and replaces the final classifier for
two output classes while retaining the pretrained feature extractor and ASPP
parameters. The model adapter normalizes torchvision's dictionary output to the
shared logits contract. Offline structural tests use `weights=None`. The
checkpoint records the resolved weight enum and torchvision version.

No model-specific branch is allowed in the trainer beyond construction through
the model registry.

## 10. Loss And Optimization

The minimal configuration uses pixelwise cross-entropy with
`ignore_index=255`. The reference U-Net configuration uses:

```text
total_loss = 0.5 * cross_entropy + 0.5 * foreground_soft_dice_loss
```

Cross-entropy includes valid background and foreground pixels. Soft Dice uses
the foreground probability after softmax and excludes ignored pixels. A small
epsilon prevents division by zero. When a batch contains no valid foreground,
the Dice term is excluded for that batch and the event is counted in logs.

The default optimizer is AdamW. The minimal configuration uses a learning rate
of `1e-3`, weight decay of `1e-4`, no scheduler, and no gradient clipping.
Reference configurations use the values in Section 6. Gradient clipping remains
configurable but disabled by default. The first release does not include AMP or
early stopping.

## 11. Training And Checkpoints

`device: auto` selects CUDA, then MPS, then CPU according to availability.
Explicitly requesting an unavailable device is an error. Reproducibility seeds
Python, NumPy, PyTorch, samplers, and data-loader workers. Device-specific
nondeterminism is documented rather than hidden.

A dry run executes data loading, model construction, forward pass, loss,
backward pass, and one optimizer update without creating a normal training run.
It prints image, mask, and logits shapes; unique mask values; valid-pixel ratio;
loss components; device; and a success marker.

Each training epoch writes metrics and updates `last.pt`. `best.pt` is selected
by validation pet foreground IoU. A checkpoint contains:

- checkpoint schema version;
- model name, constructor settings, and weights;
- class names, raw label mapping, and ignore value;
- input size and normalization;
- complete effective configuration;
- epoch, optimizer, scheduler, and best metric state;
- random number generator states;
- manifest checksums;
- package version and Git revision when available.

Resume restores all training state and rejects incompatible model, label, or
manifest changes unless a narrowly defined non-strict option is explicitly
documented. Evaluation and prediction restore only the state they require.

## 12. Evaluation

Evaluation accumulates a two-class confusion matrix over all valid pixels and
reports:

- pet foreground IoU, the primary metric;
- background IoU and mean IoU;
- pet Dice and macro Dice;
- pixel accuracy;
- valid and ignored pixel counts.

It also calculates per-image pet IoU, pet Dice, ground-truth foreground ratio,
and predicted foreground ratio. Dataset-level confusion metrics and averages of
per-image metrics are named separately.

If both prediction and target have no foreground, per-image pet IoU and Dice are
`NaN`; the sample is counted in an `empty_target_count` field and excluded from
the per-image mean. Global confusion metrics remain well defined.

Evaluation artifacts are written below the run directory:

```text
artifacts/<run_name>/
|-- best.pt
|-- config.yaml
|-- last.pt
|-- metrics.csv
|-- run.yaml
`-- evaluation/
    |-- metrics.json
    |-- per_class_metrics.csv
    |-- per_image_metrics.csv
    |-- loss_curve.png
    |-- sample_overlays/
    `-- worst_cases/
```

Worst cases are ranked by per-image pet IoU. Each rendered comparison shows the
original image, ground truth, prediction, and an error overlay. The overlay uses
distinct colors for true positive, false positive, false negative, and ignored
pixels. It includes a legend and does not treat ignored pixels as errors.

## 13. Inference

Inference requires a checkpoint and an image or input directory. It restores
the model, class mapping, and preprocessing from the checkpoint, retains each
original image size, predicts at the configured network size, and restores the
class mask with nearest-neighbor interpolation.

For each input, inference can save:

- a machine-readable single-channel PNG containing class indices 0 and 1;
- a separately named color mask;
- an alpha overlay on the original image.

Colorized outputs are never accepted as training masks by default. Single-image
and directory prediction share one implementation and preserve relative input
paths to prevent filename collisions.

## 14. Configuration And CLI

Configuration precedence is:

```text
code defaults < YAML file < --set override < dedicated CLI option
```

Configuration parsing is strict. Unknown keys, invalid types, invalid image
sizes, impossible label mappings, and incompatible model/loss combinations fail
before expensive work. `segment show-config` prints the complete effective
configuration and the origin of overridden values.

The first release exposes:

```text
segment --version
segment show-config
segment prepare-data
segment train [--dry-run]
segment evaluate
segment predict
```

Readable supporting programs remain scripts:

```text
scripts/download_data.py
scripts/preview_dataset.py
scripts/plot_metrics.py
```

Expected user errors are concise and actionable. Full tracebacks are shown only
with `--debug`. Examples include unmatched files, unexpected mask values,
unavailable explicit devices, incompatible checkpoints, and output paths that
would overwrite inputs. Checkpoint schema mismatches identify the found and
supported versions.

## 15. Testing And CI

### 15.1 Unit Tests

Unit tests cover:

- raw label mapping and ignored pixels;
- deterministic paired geometric transforms;
- nearest-neighbor mask resize preserving class values;
- cross-entropy and Dice behavior with ignored and empty masks;
- metrics against hand-calculated confusion matrices;
- U-Net output shape and skip connections;
- DeepLabV3 output adaptation without network access;
- checkpoint round trips and compatibility failures;
- strict configuration parsing and override precedence.

### 15.2 Integration And End-To-End Tests

Tests create temporary synthetic images and masks. They cover paired-directory
validation, manifest generation, dry run, one tiny CPU training epoch,
evaluation, checkpoint-only inference, and visual artifact creation. No CI test
downloads Oxford data or pretrained weights.

CLI tests cover help, successful commands, expected failure messages, and all
README commands that can run on synthetic fixtures.

### 15.3 CI Matrix

Linux CPU CI covers Python 3.10, 3.11, and 3.12. It runs Ruff, pytest, package
build, and package metadata validation. Platform-specific MPS behavior is
verified locally and documented; CI does not claim MPS coverage without an
actual Apple runner.

## 16. Documentation

The following English and Simplified Chinese documents have matching content:

```text
README
learning-path
code-tour
how-it-works
configuration-flow
config-reference
using-your-data
experiments
reference-run
tutorial/00-basics
tutorial/01-environment
tutorial/02-data-and-masks
tutorial/03-unet
tutorial/04-training
tutorial/05-evaluation-and-inference
```

The learning material begins with runnable examples and then points to the
corresponding production module. It explicitly teaches mask value semantics,
paired transforms, logits versus class masks, ignore pixels, IoU versus Dice,
validation versus test use, and visual error analysis.

Documentation never presents a reference score without the conditions needed
to reproduce and interpret it. Commands are checked by automated tests or by a
documented release verification script.

## 17. Dependencies And Packaging

The core dependency set is limited to PyTorch, torchvision, NumPy, Pillow, and
PyYAML. Matplotlib is an optional plotting dependency and part of the developer
environment. The package supports Python 3.10 through 3.12 and ships `py.typed`.

Development dependencies include pytest, Ruff, pre-commit, build, and package
metadata validation. Dependency ranges are bounded to prevent unreviewed major
version upgrades.

## 18. Delivery Sequence

Implementation should proceed in behavior-complete vertical slices:

1. Package, configuration, and quality-tool foundation.
2. Synthetic paired data, validation, manifests, and previews.
3. U-Net, minimal loss, and dry run.
4. Training, metrics, and checkpoints.
5. Evaluation, inference, and visual artifacts.
6. Oxford download and deterministic splits.
7. DeepLabV3 and the controlled loss/model experiments.
8. Bilingual learning material and reference run evidence.
9. Full release verification, repository metadata, and remote publication.

Each slice must include its focused tests and documentation updates. The final
remote repository is created and pushed only after local verification and an
explicit review of the files to publish.

## 19. Key Risks And Decisions

- CPU training may still vary widely by machine. The fixed 256/64/64 minimal
  subset and 128 x 128 input bound the default workload; dry run remains the
  fastest compatibility check.
- Mask interpolation can silently corrupt labels. All mask resize operations are
  centralized and tested for value preservation.
- Background-heavy metrics can look deceptively strong. Foreground IoU selects
  checkpoints and appears first in reports; pixel accuracy remains secondary.
- Oxford label `3` is easy to misdescribe as a semantic boundary. The project
  consistently calls it unclassified/ignored and teaches that decision.
- Pretrained weights require network access. The U-Net path is fully offline
  after dataset download, CI never downloads weights, and the DeepLab command
  reports the download requirement.
- Bilingual docs can drift. Matching document structure and command checks are
  release requirements.
- A beginner project can grow into a framework. The first-release non-goals are
  binding unless a later design explicitly changes them.

# Tutorial 01: Stabilize the Environment Before Accelerating

[简体中文](01-environment.zh-CN.md) | [Tutorial index](README.md)

The goal is a reproducible Python environment with understandable failures, not
the fastest device on the first command. Verify a CPU dry run before debugging
CUDA or Apple MPS.

## Install and identify the environment

```bash
uv sync --extra dev
uv run segment --version
uv run segment --help
```

`uv sync` creates `.venv` from `uv.lock`; `uv run` selects it without manual
activation. Confirm Python, PyTorch, and the package come from that environment:

```bash
uv run python -c "import sys, torch, image_segmenter; print(sys.version); print(torch.__version__); print(image_segmenter.__file__)"
```

The package path should point into this repository's `src/image_segmenter`.

## Inspect devices

```bash
uv run python -c "import torch; print('cuda', torch.cuda.is_available()); print('mps', torch.backends.mps.is_available())"
```

`device: auto` chooses CUDA, then MPS, then CPU. An explicitly unavailable
device raises an error instead of silently changing hardware.

After preparing data in Tutorial 02, use the bounded config for a CPU pipeline check:

```bash
uv run segment train --config configs/learning_minimal.yaml --dry-run --device cpu
```

Device kernels
may produce small floating-point differences even with fixed seeds; record the
environment and compare trends rather than expecting cross-device bit identity.

## Verify the resolved configuration

```bash
uv run segment show-config --config configs/learning_minimal.yaml
```

Look for 128 px inputs, 256/64 sample limits, two epochs, the small U-Net, and
`pretrained: false`. Then run a data-free example:

```bash
uv run python examples/01_mask_and_logits.py
```

If this works, Python, PyTorch, and the local package are connected. Later
manifest failures belong to data setup, not base installation.

## NVIDIA CUDA

Check the driver and the PyTorch build before changing training settings:

```bash
nvidia-smi
uv run python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
uv run segment train --config configs/learning_minimal.yaml --dry-run --device cuda --set train.amp true
```

Run the CUDA dry run only after `prepare-data` has created `data/manifests/dataset.yaml`.

AMP is effective only on CUDA. Its scaler state is saved in checkpoints, so do not change the
effective AMP setting while resuming one run. CPU and MPS use full precision even when AMP is
requested and print that decision. This project uses one device and does not implement multi-GPU
or distributed training.

For out-of-memory errors, reduce batch size first, then image size, then U-Net width or encoder
size. Model-specific memory use differs, so a batch size that works for LRASPP may not work for FCN.

## Apple MPS

On Apple silicon, confirm `torch.backends.mps.is_available()` and use `--device mps`. MPS shares
system memory and does not use this project's AMP path. Start with `num_workers: 0`; after a dry run,
increase workers or batch size one setting at a time. Return to `--device cpu` when diagnosing an
operation that is unsupported by the installed PyTorch MPS backend.

## Kaggle GPU notebook

In Notebook settings, select a GPU accelerator before starting the session. The following cells
keep source and artifacts under writable `/kaggle/working`; attached datasets remain read-only under
`/kaggle/input`.

```bash
!nvidia-smi
%cd /kaggle/working
!git clone https://github.com/Doithoo/pytorch-image-segmentation-lab.git
%cd /kaggle/working/pytorch-image-segmentation-lab
!pip install -q uv
!uv sync --locked --extra dev
```

Replace `my-segmentation-data` with the attached Kaggle dataset directory. This example expects
paired `images/` and `masks/` folders:

```bash
!uv run segment prepare-data --config configs/learning_minimal.yaml \
  --data-dir /kaggle/input/my-segmentation-data \
  --manifest-dir /kaggle/working/manifests --source-format paired

!uv run segment train --config configs/learning_minimal.yaml --dry-run --device cuda \
  --output-dir /kaggle/working/artifacts \
  --set data.manifest_dir /kaggle/working/manifests --set train.amp true

!uv run segment train --config configs/learning_minimal.yaml --device cuda \
  --output-dir /kaggle/working/artifacts --set run_name kaggle-check \
  --set data.manifest_dir /kaggle/working/manifests \
  --set train.epochs 1 --set train.amp true
```

Resume only from the same run directory and keep the effective AMP setting unchanged:

```bash
!uv run segment train --config configs/learning_minimal.yaml --device cuda \
  --output-dir /kaggle/working/artifacts --set run_name kaggle-check \
  --set data.manifest_dir /kaggle/working/manifests \
  --set train.epochs 2 --set train.amp true \
  --resume /kaggle/working/artifacts/kaggle-check/last.pt

!uv run segment evaluate \
  --checkpoint /kaggle/working/artifacts/kaggle-check/best.pt \
  --split valid --device cuda \
  --output-dir /kaggle/working/artifacts/kaggle-check/evaluation
!tar -czf /kaggle/working/kaggle-check-artifacts.tgz -C /kaggle/working/artifacts kaggle-check
```

Download the archive from the notebook Output panel before ending the session. Kaggle availability,
hardware, and session policies can change; these commands are documented but were not hardware-run
as part of the local macOS verification.

For CUDA/MPS failures, worker hangs, or OOM, return to CPU and `num_workers: 0`,
then change one setting at a time. Never disable system memory safety limits to
force a run. See [troubleshooting](../guides/troubleshooting.md), then continue to
[data and masks](02-data-and-masks.md).

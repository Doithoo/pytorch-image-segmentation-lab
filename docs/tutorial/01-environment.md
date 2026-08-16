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

```bash
uv run segment train --config configs/learning_minimal.yaml --dry-run --device cpu
```

The minimal config is bounded enough for a CPU pipeline check. Device kernels
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

For CUDA/MPS failures, worker hangs, or OOM, return to CPU and `num_workers: 0`,
then change one setting at a time. Never disable system memory safety limits to
force a run. See [troubleshooting](../troubleshooting.md), then continue to
[data and masks](02-data-and-masks.md).

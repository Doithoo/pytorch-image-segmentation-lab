# Tutorial 01: Environment

[简体中文](01-environment.zh-CN.md) | [Tutorial index](../learning-path.md)

The project uses uv to pin and install one reproducible Python environment.

## Install

Use Python 3.10-3.12. From the repository root:

```bash
uv sync --extra dev
uv run segment --version
uv run segment --help
```

`uv run` executes inside `.venv`; manual activation is optional. The dev extra
contains tests, lint, build, metadata validation, and plotting dependencies.

## Check PyTorch And Devices

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.backends.mps.is_available()); print(torch.cuda.is_available())"
```

`device: auto` prefers CUDA, then Apple MPS, then CPU. Explicitly requesting an
unavailable accelerator fails. Start with CPU for learning and debugging;
change hardware only after the workflow is correct.

## Inspect Configuration

```bash
uv run segment show-config --config configs/learning_minimal.yaml
```

Confirm Python can import the installed editable package and that the output
shows a CPU-bounded subset, 128 px image size, and two epochs.

## Troubleshoot Deliberately

If installation fails, record the Python version, operating system, command,
and first error. Do not mix pip-installed packages into the managed environment
until the cause is understood. Use `--debug` only when a normal CLI error lacks
enough detail.

Next: [Tutorial 02](02-data-and-masks.md).

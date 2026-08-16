# Configuration Flow

[简体中文](configuration-flow.zh-CN.md) | [Back to README](../README.md)

Configuration is explicit and strict. It is resolved once before data loading
or model construction.

## Precedence

```text
code defaults < YAML file < --set override < dedicated CLI option
```

A later layer replaces an earlier value. Dedicated flags such as `--device`
are best for one-off runtime choices; YAML should describe a reproducible
experiment; `--set` is useful for a controlled variation.

## Inspect Before Running

```bash
uv run segment show-config --config configs/learning_minimal.yaml
uv run segment show-config --config configs/learning_minimal.yaml \
  --set train.epochs 3 --set model.base_channels 8
```

The first command prints every effective field. The second demonstrates typed
dotted overrides. Unknown paths, invalid booleans, non-divisible image sizes,
and incompatible label mappings fail immediately.

## Command-Specific Overrides

```bash
uv run segment train --config configs/learning_minimal.yaml \
  --set train.epochs 3 --device cpu --output-dir artifacts
```

Here `--device cpu` and `--output-dir artifacts` have the highest precedence.
The saved `config.yaml` is the resolved result, not merely the input YAML.

## Checkpoint Restoration

Resume validates model construction, raw label mapping, ignore index, and
train/validation manifest hashes before loading optimizer state. Evaluation and
prediction reconstruct the model and preprocessing from checkpoint metadata;
they do not depend on today's config defaults.

See [Config Reference](config-reference.md) for field meanings and
[Experiments](experiments.md) for safe comparisons.

# Configuration Flow

[简体中文](configuration-flow.zh-CN.md) | [Documentation](../README.md)

One resolved `ExperimentConfig` is created and validated before data or models:

```text
code defaults < YAML < --set < dedicated flags
```

```bash
uv run segment show-config --config configs/learning_minimal.yaml \
  --set train.epochs 5 --device cpu
```

`--set` parses JSON values to preserve numbers, booleans, null, lists, and maps.
`show-config` prints exactly what the command will use; training saves that
resolved result as `config.yaml`.

Validation rejects unknown keys, wrong types, invalid labels/loss/model/device,
and other inconsistent settings before expensive work. Sample-count-dependent
empty splits are detected by `prepare-data` after scanning.

Checkpoint evaluation and prediction restore model/preprocessing from the saved
config. Training resume additionally compares model, data, loss, batch, labels,
and manifests. A cosine checkpoint must keep its planned total epochs. See the
[configuration reference](../reference/config-reference.md).

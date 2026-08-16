# Test Guide

[简体中文](README.zh-CN.md) | [Contributing](../CONTRIBUTING.md)

Tests use tiny synthetic fixtures. They must not download Oxford, pretrained
weights, or any network resource.

## Run Tests

```bash
uv run pytest
uv run pytest tests/test_end_to_end.py -v
```

Use a focused file during red-green development, then run the full suite before
committing. Release verification treats deprecation warnings as errors.

## Test Layers

Unit tests cover label mapping, paired transforms, loss, metrics, U-Net shapes,
config, and checkpoint schemas. Integration tests cover manifests, CLI,
evaluation, inference, scripts, and packaging. The end-to-end test prepares
synthetic pairs, trains on CPU, evaluates, and predicts.

## Fixtures

`paired_data` creates aligned RGB images and Oxford-style masks. `oxford_data`
adds synthetic official split files. `trained_run` creates a tiny real
checkpoint. Prefer these behavioral fixtures over mocks.

## Adding Coverage

Write a focused failing test first. Assert observable files, values, shapes, or
errors rather than private call counts. Fix warnings instead of filtering them.
Keep expensive data and model downloads outside CI.

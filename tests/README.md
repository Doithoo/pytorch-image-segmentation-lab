# Tests as Executable Behavior

[简体中文](README.zh-CN.md) | [Contributing](../CONTRIBUTING.md)

Learners need not read every test, but small inputs explain edge cases well.
Start with metrics, dataset mapping, examples, then end-to-end.

```bash
uv run pytest tests/test_metrics.py -v
uv run pytest tests/test_end_to_end.py -v
uv run pytest
```

Tests use synthetic data and never download Oxford or pretrained weights. For a
behavior change, write a focused failing test, confirm the reason, implement the
smallest fix, then run the full suite. Assert observable files, values, shapes,
and errors rather than private calls. Reuse `paired_data`, `oxford_data`, and
`trained_run` fixtures before introducing mocks.

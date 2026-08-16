# Contributing

[简体中文](CONTRIBUTING.zh-CN.md)

Ask whether a change helps a learner run, observe, or understand the project
while preserving inspectable experiments. Technically correct but opaque
features may need a smaller first version.

## Before coding

Open an issue for behavior or architecture changes. Describe the learner
problem, expected result, alternatives, data/network impact, and verification.
Do not commit Oxford data, downloaded model weights, secrets, or full local run
directories.

## Local checks

```bash
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
uv build
uv run twine check dist/*
```

Write a failing test before production behavior. Keep tests offline with
synthetic fixtures and `weights=None`; do not bypass hooks.

## Documentation

English and Simplified Chinese core documents both need discoverable entries and
the same behavior. They may organize headings, examples, and troubleshooting in
ways natural to each language. Commands, configuration names, labels, and metric
definitions must agree. Publish measured values only with conditions and limits.

Store model binaries in a release/model artifact service. Keep configs, hashes,
metrics, and a few representative images in Git.

## Pull requests

Explain the learner problem, change, checks run, artifact differences, and known
limits. Separate unrelated refactors and use English Conventional Commits.

Contributions are released under the [MIT License](LICENSE).

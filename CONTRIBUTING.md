# Contributing

[简体中文](CONTRIBUTING.zh-CN.md)

Contributions should preserve the project's beginner-readable behavior,
reproducibility contract, and English/Simplified Chinese parity.

## Before Coding

Open an issue for behavior or architecture changes. State the learner problem,
expected behavior, alternatives, data/network implications, and how it will be
tested. Never add Oxford data, downloaded model weights, or secrets to Git.

## Local Workflow

```bash
uv sync --extra dev
uv run ruff check .
uv run pytest
uv build
uv run twine check dist/*
```

Write a failing test before production behavior. Keep commits focused and use
English Conventional Commits. Do not bypass hooks. Tests and CI must remain
offline: use synthetic fixtures and `weights=None`.

## Documentation

Update the English and `.zh-CN.md` counterpart together with identical heading
levels and shell command blocks. Explain new fields in the config reference and
link learning examples to their production module. Never publish an unmeasured
metric or omit the conditions required to interpret it.

## Pull Requests

Describe what changed, why it helps learners, tests run, artifact or UI changes,
and known limitations. Keep unrelated refactors separate. Maintainers may ask
for smaller examples or clearer naming even when code is technically correct.

By contributing, you agree that your work is released under the repository's
[MIT License](LICENSE).

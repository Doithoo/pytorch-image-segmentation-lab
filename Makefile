.PHONY: lint format typecheck test test-model-zoo build metadata check

lint:
	uv run ruff check .

format:
	uv run ruff format --check .

typecheck:
	uv run mypy

test:
	uv run pytest

test-model-zoo:
	uv run --extra model-zoo pytest -m model_zoo tests/test_smp_models.py

build:
	uv run python -m build

metadata:
	uv run twine check dist/*

check: lint format typecheck test build metadata

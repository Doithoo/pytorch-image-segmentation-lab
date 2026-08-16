.PHONY: lint format typecheck test build metadata check

lint:
	uv run ruff check .

format:
	uv run ruff format --check .

typecheck:
	uv run mypy

test:
	uv run pytest

build:
	uv run python -m build

metadata:
	uv run twine check dist/*

check: lint format typecheck test build metadata

.PHONY: lint test build metadata check

lint:
	uv run ruff check .

test:
	uv run pytest

build:
	uv run python -m build

metadata:
	uv run twine check dist/*

check: lint test build metadata

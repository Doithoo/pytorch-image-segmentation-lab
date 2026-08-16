.PHONY: lint test build

lint:
	uv run ruff check .

test:
	uv run pytest

build:
	uv run python -m build

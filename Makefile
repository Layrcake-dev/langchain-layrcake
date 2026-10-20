.PHONY: lint format typecheck test live build check

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run mypy

test:
	uv run pytest tests/unit_tests --disable-socket --allow-unix-socket

live:
	uv run pytest -m live tests/integration_tests

build:
	rm -rf dist
	uv build

check: lint typecheck test

.PHONY: setup run-tests run-project extract-data check format

setup:
	uv sync --group dev

run-tests:
	PYTHONPATH=src uv run python -m unittest discover -s tests -v

run-project:
	uv run python src/main.py

extract-data:
	uv run python scripts/extract_m5.py

check:
	uv run ruff check src scripts tests

format:
	uv run ruff format src scripts tests

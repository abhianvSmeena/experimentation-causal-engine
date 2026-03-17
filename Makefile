PY := .venv/bin/python

.PHONY: setup data test lint asos criteo all

setup:
	uv venv --python 3.12
	uv pip install -e ".[case-studies,dev]"

data:
	$(PY) scripts/download_data.py

test:
	.venv/bin/pytest

lint:
	.venv/bin/ruff check . && .venv/bin/ruff format --check .

asos:
	$(PY) case_studies/asos/run_all.py

criteo:
	$(PY) case_studies/criteo/prepare.py
	$(PY) case_studies/criteo/run_all.py

all: data test asos criteo

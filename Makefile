.PHONY: install lint format test check data eval traces erp deploy destroy

PYTHON ?= python

install:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

test:
	$(PYTHON) -m pytest

check: lint test

data:
	$(PYTHON) scripts/tasks.py data

eval:
	$(PYTHON) scripts/tasks.py eval --version v1

traces:
	$(PYTHON) scripts/tasks.py traces --version v1

erp:
	$(PYTHON) scripts/tasks.py erp

deploy:
	$(PYTHON) scripts/tasks.py deploy

destroy:
	$(PYTHON) scripts/tasks.py destroy

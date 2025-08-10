# Variables (prefer project venv if present)
PYTHON ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || command -v python3)
PIP := $(PYTHON) -m pip
APP := app.main:app
IMAGE ?= stack-app
TAG ?= latest
PORT ?= 8000

.PHONY: help venv install upgrade fmt lint typecheck test test-cov check run dev docker-build docker-run clean

help:
	@echo "Available targets:"
	@echo "  venv        - Create a local virtualenv at .venv"
	@echo "  install     - Install prod and dev dependencies (uses $(PYTHON))"
	@echo "  upgrade     - Upgrade pip/setuptools/wheel"
	@echo "  fmt         - Auto-format code (black + isort + ruff --fix)"
	@echo "  lint        - Lint (ruff) and enforce formatting (black/isort --check)"
	@echo "  typecheck   - Static type check with mypy"
	@echo "  test        - Run tests"
	@echo "  test-cov    - Run tests with coverage"
	@echo "  check       - Run lint, typecheck, and tests"
	@echo "  run         - Run uvicorn server"
	@echo "  dev         - Run uvicorn with reload for development"
	@echo "  docker-build- Build docker image"
	@echo "  docker-run  - Run dockerized app"
	@echo "  clean       - Remove Python caches and build artifacts"

venv:
	python3 -m venv .venv

install: | venv
	$(PIP) install -U pip setuptools wheel
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt

upgrade:
	$(PIP) install -U pip setuptools wheel

fmt:
	$(PYTHON) -m ruff check . --fix
	$(PYTHON) -m isort . --profile black
	$(PYTHON) -m black .

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m black . --check
	$(PYTHON) -m isort . --check-only --profile black

typecheck:
	$(PYTHON) -m mypy app --ignore-missing-imports --install-types --non-interactive

test:
	$(PYTHON) -m pytest -q

test-cov:
	$(PYTHON) -m pytest --cov=app --cov-report=term-missing

check: lint typecheck test

run:
	$(PYTHON) -m uvicorn $(APP) --host 0.0.0.0 --port $(PORT)

dev:
	$(PYTHON) -m uvicorn $(APP) --host 0.0.0.0 --port $(PORT) --reload

docker-build:
	docker build -t $(IMAGE):$(TAG) .

docker-run:
	docker run --rm -p $(PORT):8000 -e ENV=prod -e LOG_LEVEL=INFO $(IMAGE):$(TAG)

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .mypy_cache .pytest_cache .ruff_cache .coverage



.PHONY: help setup install install-frontend lint format test train api frontend docker-up docker-down clean

help:
	@echo "Targets:"
	@echo "  setup           Create venv + install Python deps + frontend deps + pre-commit"
	@echo "  install         pip install -r requirements.txt"
	@echo "  install-frontend  npm ci in frontend/"
	@echo "  lint            Ruff check + frontend lint"
	@echo "  format          Ruff format"
	@echo "  test            Run pytest"
	@echo "  train           Run full training pipeline (src/main.py)"
	@echo "  api             Run FastAPI dev server on :8000"
	@echo "  frontend        Run Vite dev server on :5173"
	@echo "  docker-up       Build + start the full docker compose stack"
	@echo "  docker-down     Stop docker compose stack"
	@echo "  clean           Remove generated outputs (preserves curated docs/figures/)"

setup:
	python3 -m venv venv
	. venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt && pip install pre-commit ruff pytest nbstripout
	. venv/bin/activate && pre-commit install
	cd frontend && npm ci

install:
	pip install -r requirements.txt

install-frontend:
	cd frontend && npm ci

lint:
	ruff check .
	cd frontend && npm run lint

format:
	ruff format .

test:
	PYTHONPATH=. pytest tests/ -v

train:
	python3 src/main.py

api:
	python3 -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	cd frontend && npm run dev

docker-up:
	docker compose -f docker/docker-compose.yml build
	docker compose -f docker/docker-compose.yml up -d

docker-down:
	docker compose -f docker/docker-compose.yml down

clean:
	find outputs -type f ! -path 'outputs/.gitkeep' -delete 2>/dev/null || true
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true

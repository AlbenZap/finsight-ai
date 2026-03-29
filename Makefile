.PHONY: dev dev-ollama dev-backend test lint lint-fix build build-hf ollama-pull help

# ── Local development ─────────────────────────────────────────────────────────
dev:
	@echo "Starting FinSight AI backend (NVIDIA NIM mode)..."
	docker compose up --build backend

dev-ollama:
	@echo "Starting FinSight AI backend + Ollama (self-hosted mode)..."
	docker compose --profile ollama up --build

dev-backend:
	docker compose up --build backend

# ── Testing ───────────────────────────────────────────────────────────────────
test:
	docker compose run --rm backend pytest tests/ -v --cov=. --cov-report=term-missing

# ── Linting ───────────────────────────────────────────────────────────────────
lint:
	docker compose run --rm backend ruff check .
	docker compose run --rm backend ruff format --check .

lint-fix:
	docker compose run --rm backend ruff check --fix .
	docker compose run --rm backend ruff format .

# ── Build smoke test (single Dockerfile, local port 8000) ────────────────────
build:
	docker build -t finsight-backend .

# ── Build HF Spaces image (same Dockerfile, default PORT=7860) ───────────────
build-hf:
	docker build -t finsight-hf .

# ── Ollama (Docker) ───────────────────────────────────────────────────────────
ollama-pull:
	docker exec finsight-ollama ollama pull llama3.1:8b

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo "FinSight AI - Available commands:"
	@echo "  make dev          Start backend only (NVIDIA NIM mode, default)"
	@echo "  make dev-ollama   Start backend + Ollama (self-hosted LLM mode)"
	@echo "  make test         Run pytest with coverage"
	@echo "  make lint         Run Ruff linter"
	@echo "  make build        Build backend Docker image (smoke test)"
	@echo "  make build-hf     Build HF Spaces Docker image (smoke test)"
	@echo "  make ollama-pull  Pull Llama 3.1 8B into Ollama container (first-time)"
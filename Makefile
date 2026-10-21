.PHONY: test lint format benchmark benchmark-llm server dashboard compiler clean help

# Default target
help: ## Show this help message
	@echo "╔══════════════════════════════════════════════════════╗"
	@echo "║            zGPU — Development Commands              ║"
	@echo "╚══════════════════════════════════════════════════════╝"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

PYTHONPATH := PYTHONPATH=$(CURDIR)

# ──────────────────────── Testing ────────────────────────

test: ## Run the full pytest test suite
	$(PYTHONPATH) python3 -m pytest tests/ -v --tb=short

test-fast: ## Run tests without verbose output
	$(PYTHONPATH) python3 -m pytest tests/ -q

test-cov: ## Run tests with coverage report
	$(PYTHONPATH) python3 -m pytest tests/ -v --tb=short --cov=. --cov-report=html

# ──────────────────────── Linting ────────────────────────

lint: ## Run ruff linter
	ruff check .

format: ## Format code with ruff
	ruff format .

format-check: ## Check formatting without making changes
	ruff format --check .

check: lint format-check test ## Run all checks (lint + format + test)

# ──────────────────────── Benchmarks ────────────────────────

benchmark: ## Run silicon benchmarks (vector add + tensor MMA)
	$(PYTHONPATH) python3 benchmarks/run_benchmarks.py

benchmark-llm: ## Run LLM acceleration benchmark
	$(PYTHONPATH) python3 benchmarks/benchmark_llm.py

benchmark-cpu: ## Run network CPU powerup benchmark
	$(PYTHONPATH) python3 benchmarks/benchmark_cpu_powerup.py

benchmark-agi: ## Run SuperAGI benchmark
	$(PYTHONPATH) python3 benchmarks/benchmark_super_agi.py

benchmark-all: benchmark benchmark-llm benchmark-cpu benchmark-agi ## Run all benchmarks

# ──────────────────────── Services ────────────────────────

server: ## Start LLM JSON-RPC inference server (port 8880)
	$(PYTHONPATH) python3 llm/server.py 8880

dashboard: ## Start telemetry dashboard (port 9000)
	$(PYTHONPATH) python3 dashboard/server.py 9000

compiler: ## Start web compiler studio (port 9100)
	$(PYTHONPATH) python3 compiler/service.py 9100

worker: ## Start CPU worker node (port 9200)
	$(PYTHONPATH) python3 cluster/worker/cpu_worker.py 9200

# ──────────────────────── Build ────────────────────────

build: ## Build the Python package
	python3 -m build

install-dev: ## Install in development mode with dev dependencies
	pip install -e ".[dev]"

# ──────────────────────── Clean ────────────────────────

clean: ## Remove build artifacts and caches
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ htmlcov/ .coverage .ruff_cache/

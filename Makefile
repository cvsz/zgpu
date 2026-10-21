.PHONY: test test-fast test-cov lint format format-check check benchmark benchmark-llm benchmark-cpu benchmark-agi benchmark-all server dashboard compiler worker start stop restart status logs-server logs-dashboard logs-compiler logs-worker build compile-sample install-dev clean help

PID_DIR := .pids
LOG_DIR := .logs
PYTHONPATH_ENV := PYTHONPATH=$(CURDIR)

# Default target
help: ## Show this help message
	@echo "╔════════════════════════════════════════════════════════════════════╗"
	@echo "║                   zGPU — Control & Dev Makefile                    ║"
	@echo "╚════════════════════════════════════════════════════════════════════╝"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ──────────────────────── Service Control (Daemon Management) ────────────────────────

start: ## Start all background services (LLM, Dashboard, Compiler, CPU Worker)
	@./scripts/start_services.sh

stop: ## Stop all background services
	@./scripts/stop_services.sh

restart: stop start ## Restart all background services

status: ## Show running status and health of all services
	@echo "════════════════════════════════════════════════════════════════════"
	@echo "                  zGPU Service Health & Ports                       "
	@echo "════════════════════════════════════════════════════════════════════"
	@printf "  %-20s %-8s %-12s %s\n" "SERVICE" "PORT" "STATUS" "ENDPOINT"
	@printf "  %-20s %-8s %-12s %s\n" "--------------------" "--------" "------------" "-------------------------"
	@curl -s -X POST http://127.0.0.1:8880/rpc -H "Content-Type: application/json" -d '{"jsonrpc": "2.0", "id": 1, "method": "system.health"}' 2>/dev/null | grep -qE '"healthy":\s*true' \
		&& printf "  \033[32m%-20s\033[0m %-8s \033[32m%-12s\033[0m %s\n" "LLM Server" "8880" "RUNNING" "http://localhost:8880/rpc" \
		|| printf "  \033[31m%-20s\033[0m %-8s \033[31m%-12s\033[0m %s\n" "LLM Server" "8880" "DOWN" "http://localhost:8880/rpc"
	@curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:9000/api/telemetry 2>/dev/null | grep -q "200" \
		&& printf "  \033[32m%-20s\033[0m %-8s \033[32m%-12s\033[0m %s\n" "Dashboard" "9000" "RUNNING" "http://localhost:9000/" \
		|| printf "  \033[31m%-20s\033[0m %-8s \033[31m%-12s\033[0m %s\n" "Dashboard" "9000" "DOWN" "http://localhost:9000/"
	@curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:9100/ 2>/dev/null | grep -q "200" \
		&& printf "  \033[32m%-20s\033[0m %-8s \033[32m%-12s\033[0m %s\n" "Compiler Studio" "9100" "RUNNING" "http://localhost:9100/" \
		|| printf "  \033[31m%-20s\033[0m %-8s \033[31m%-12s\033[0m %s\n" "Compiler Studio" "9100" "DOWN" "http://localhost:9100/"
	@curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:9200/health 2>/dev/null | grep -q "200" \
		&& printf "  \033[32m%-20s\033[0m %-8s \033[32m%-12s\033[0m %s\n" "CPU Worker" "9200" "RUNNING" "http://localhost:9200/health" \
		|| printf "  \033[31m%-20s\033[0m %-8s \033[31m%-12s\033[0m %s\n" "CPU Worker" "9200" "DOWN" "http://localhost:9200/health"
	@echo "════════════════════════════════════════════════════════════════════"

# ──────────────────────── Foreground Server Runners ────────────────────────

server: ## Run LLM server in foreground (port 8880)
	$(PYTHONPATH_ENV) python3 llm/server.py 8880

dashboard: ## Run telemetry dashboard in foreground (port 9000)
	$(PYTHONPATH_ENV) python3 dashboard/server.py 9000

compiler: ## Run web compiler studio in foreground (port 9100)
	$(PYTHONPATH_ENV) python3 compiler/service.py 9100

worker: ## Run CPU worker node in foreground (port 9200)
	$(PYTHONPATH_ENV) python3 cluster/worker/cpu_worker.py 9200

# ──────────────────────── Logs ────────────────────────

logs-server: ## Tail LLM server logs
	tail -f $(LOG_DIR)/server.log

logs-dashboard: ## Tail telemetry dashboard logs
	tail -f $(LOG_DIR)/dashboard.log

logs-compiler: ## Tail compiler studio logs
	tail -f $(LOG_DIR)/compiler.log

logs-worker: ## Tail CPU worker logs
	tail -f $(LOG_DIR)/worker.log

# ──────────────────────── Testing ────────────────────────

test: ## Run the full pytest test suite
	$(PYTHONPATH_ENV) python3 -m pytest tests/ -v --tb=short

test-fast: ## Run tests without verbose output
	$(PYTHONPATH_ENV) python3 -m pytest tests/ -q

test-cov: ## Run tests with coverage report
	$(PYTHONPATH_ENV) python3 -m pytest tests/ -v --tb=short --cov=. --cov-report=html

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
	$(PYTHONPATH_ENV) python3 benchmarks/run_benchmarks.py

benchmark-llm: ## Run LLM acceleration benchmark
	$(PYTHONPATH_ENV) python3 benchmarks/benchmark_llm.py

benchmark-cpu: ## Run network CPU powerup benchmark
	$(PYTHONPATH_ENV) python3 benchmarks/benchmark_cpu_powerup.py

benchmark-agi: ## Run SuperAGI benchmark
	$(PYTHONPATH_ENV) python3 benchmarks/benchmark_super_agi.py

benchmark-all: benchmark benchmark-llm benchmark-cpu benchmark-agi ## Run all benchmarks

# ──────────────────────── Build & Compilation ────────────────────────

build: ## Build the Python package (sdist & wheel)
	python3 -m build

compile-sample: ## Test-compile a sample zISA assembly kernel
	@$(PYTHONPATH_ENV) python3 -c '\
from isa.assembler import Assembler;\
src = "GET.TID R0\nVADD R1, R0, R0\nHALT";\
code = Assembler.assemble(src);\
print("Sample Compiled Bytecode Words:", [hex(w) for w in code])'

install-dev: ## Install in development mode with dev dependencies
	pip install --break-system-packages -e ".[dev]" 2>/dev/null || pip install -e ".[dev]"

# ──────────────────────── Clean ────────────────────────

clean: stop ## Stop services and clean temporary build/test artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ htmlcov/ .coverage .ruff_cache/ $(PID_DIR) $(LOG_DIR) start_services.sh

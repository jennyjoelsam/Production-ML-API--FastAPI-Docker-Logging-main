.PHONY: help install install-dev run run-dev test test-cov docker-build docker-run docker-stop clean lint format docs

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt

install-dev: ## Install dependencies including dev/test dependencies
	pip install -r requirements.txt
	pip install pytest pytest-cov pytest-asyncio flake8 mypy pylint black

run: ## Run the API server
	uvicorn app.main:app --host 0.0.0.0 --port 8000

run-dev: ## Run the API server with auto-reload
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test: ## Run tests
	pytest tests/ -v

test-cov: ## Run tests with coverage report
	pytest tests/ -v --cov=app --cov-report=html --cov-report=term-missing

test-watch: ## Run tests in watch mode (requires pytest-watch)
	ptw tests/ -v

lint: ## Run linters (flake8, mypy)
	flake8 app tests --max-line-length=120
	mypy app --ignore-missing-imports

format: ## Format code with black
	black app tests

docker-build: ## Build Docker image
	docker build -t ml-inference-api:latest .

docker-run: ## Run Docker container
	docker run -p 8000:8000 ml-inference-api:latest

docker-stop: ## Stop running Docker containers
	docker stop $$(docker ps -q --filter 'ancestor=ml-inference-api:latest') || true

compose-up: ## Start services with docker-compose
	docker-compose up

compose-down: ## Stop services with docker-compose
	docker-compose down

compose-logs: ## View docker-compose logs
	docker-compose logs -f

clean: ## Clean up generated files and directories
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
	rm -rf .pytest_cache htmlcov .coverage .mypy_cache
	rm -rf dist build *.egg-info

.DEFAULT_GOAL := help

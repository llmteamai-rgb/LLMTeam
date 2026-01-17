# LLMTeam Makefile
# Enterprise AI Workflow Runtime

.PHONY: install install-dev test test-cov test-fast test-module lint format type-check build clean docs release help

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python
PYTEST := pytest
SRC := src/llmteam
TESTS := tests

#------------------------------------------------------------------------------
# Installation
#------------------------------------------------------------------------------

install: ## Install package
	pip install -e .

install-dev: ## Install package with dev dependencies
	pip install -e ".[dev]"

#------------------------------------------------------------------------------
# Testing
#------------------------------------------------------------------------------

test: ## Run all tests (sequential, safest)
	$(PYTHON) run_tests.py

test-parallel: ## Run tests with limited parallelism
	$(PYTHON) run_tests.py --parallel 2

test-cov: ## Run tests with coverage report
	$(PYTHON) run_tests.py --coverage

test-fast: ## Run unit tests only (skip integration)
	$(PYTHON) run_tests.py --fast

test-module: ## Run single module tests (usage: make test-module MODULE=canvas)
	$(PYTHON) run_tests.py --module $(MODULE)

#------------------------------------------------------------------------------
# Code Quality
#------------------------------------------------------------------------------

lint: ## Run linter (ruff)
	ruff check $(SRC) $(TESTS)

lint-fix: ## Run linter with auto-fix
	ruff check --fix $(SRC) $(TESTS)

format: ## Format code with black
	black $(SRC) $(TESTS)

format-check: ## Check code formatting without changes
	black --check $(SRC) $(TESTS)

type-check: ## Run type checker (mypy)
	mypy $(SRC)

check: lint type-check format-check ## Run all checks (lint, type, format)

#------------------------------------------------------------------------------
# Build & Release
#------------------------------------------------------------------------------

build: clean ## Build distribution packages
	$(PYTHON) -m build

clean: ## Clean build artifacts
	rm -rf dist/ build/ *.egg-info/ .eggs/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf htmlcov/ .coverage coverage.xml

release: clean check test build ## Full release: clean, check, test, build
	@echo "Ready for release. Run 'twine upload dist/*' to publish."

#------------------------------------------------------------------------------
# Documentation
#------------------------------------------------------------------------------

docs: ## Build documentation
	cd docs && make html

docs-serve: ## Serve documentation locally
	cd docs/_build/html && $(PYTHON) -m http.server 8080

#------------------------------------------------------------------------------
# Development Utilities
#------------------------------------------------------------------------------

verify: ## Verify installation
ifeq ($(OS),Windows_NT)
	@set PYTHONPATH=src && $(PYTHON) -c "import llmteam; print(f'llmteam v{llmteam.__version__}')"
else
	@PYTHONPATH=src $(PYTHON) -c "import llmteam; print(f'llmteam v{llmteam.__version__}')"
endif

shell: ## Start Python shell with llmteam imported
ifeq ($(OS),Windows_NT)
	@set PYTHONPATH=src && $(PYTHON) -i -c "import llmteam; print(f'llmteam v{llmteam.__version__} loaded')"
else
	@PYTHONPATH=src $(PYTHON) -i -c "import llmteam; print(f'llmteam v{llmteam.__version__} loaded')"
endif

#------------------------------------------------------------------------------
# Help
#------------------------------------------------------------------------------

help: ## Show this help message
	@echo "LLMTeam Development Commands"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Examples:"
	@echo "  make install-dev      Install with dev dependencies"
	@echo "  make test             Run all tests"
	@echo "  make test-module MODULE=canvas  Run canvas tests only"
	@echo "  make check            Run all quality checks"

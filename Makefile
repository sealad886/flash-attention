
# =============================================================================
# Flash Attention Makefile
# =============================================================================

.PHONY: help clean_dist create_dist upload_package \
        test test-mlx test-mlx-quick test-mlx-full test-cuda \
        benchmark-mlx install-dev install-hooks lint

# Default target
help:
	@echo "Flash Attention Development Commands"
	@echo "====================================="
	@echo ""
	@echo "Testing:"
	@echo "  make test            - Run all tests (auto-detects available backends)"
	@echo "  make test-mlx        - Run MLX backend tests (Apple Silicon only)"
	@echo "  make test-mlx-quick  - Quick MLX smoke test (~5 seconds)"
	@echo "  make test-mlx-full   - Full MLX test suite with coverage"
	@echo "  make test-cuda       - Run CUDA backend tests"
	@echo ""
	@echo "Benchmarks:"
	@echo "  make benchmark-mlx   - Run MLX benchmarks (Apple Silicon only)"
	@echo ""
	@echo "Development:"
	@echo "  make install-dev     - Install package in development mode"
	@echo "  make install-hooks   - Install git pre-push hook for MLX testing"
	@echo "  make lint            - Run linters"
	@echo ""
	@echo "Packaging:"
	@echo "  make clean_dist      - Clean distribution files"
	@echo "  make create_dist     - Create source distribution"
	@echo "  make upload_package  - Upload to PyPI"

# =============================================================================
# Testing Targets
# =============================================================================

# Run all available tests
test:
	pytest -v

# MLX Backend Tests (Apple Silicon only)
test-mlx:
	@if [ "$$(uname -s)" != "Darwin" ] || [ "$$(uname -m)" != "arm64" ]; then \
		echo "⚠️  MLX tests require Apple Silicon Mac"; \
		exit 1; \
	fi
	pytest flash_attn/flash_attn_mlx/tests/ -v --tb=short

# Quick MLX smoke test - fast validation before pushing
test-mlx-quick:
	@if [ "$$(uname -s)" != "Darwin" ] || [ "$$(uname -m)" != "arm64" ]; then \
		echo "⚠️  MLX tests require Apple Silicon Mac"; \
		exit 0; \
	fi
	@echo "🚀 Running quick MLX smoke test..."
	@python scripts/test_mlx_quick.py

# Full MLX test suite with verbose output
test-mlx-full:
	@if [ "$$(uname -s)" != "Darwin" ] || [ "$$(uname -m)" != "arm64" ]; then \
		echo "⚠️  MLX tests require Apple Silicon Mac"; \
		exit 1; \
	fi
	pytest flash_attn/flash_attn_mlx/tests/ tests/test_flash_attn_mlx.py -v --tb=long -x

# CUDA tests
test-cuda:
	pytest tests/ -v --ignore=tests/test_flash_attn_mlx.py

# =============================================================================
# Benchmarks
# =============================================================================

benchmark-mlx:
	@if [ "$$(uname -s)" != "Darwin" ] || [ "$$(uname -m)" != "arm64" ]; then \
		echo "⚠️  MLX benchmarks require Apple Silicon Mac"; \
		exit 1; \
	fi
	python benchmarks/benchmark_flash_attn_mlx.py

# =============================================================================
# Development Setup
# =============================================================================

# Install in development mode
install-dev:
	FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE pip install -e ".[dev]" 2>/dev/null || \
	FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE pip install -e .

# Install git hooks
install-hooks:
	@echo "Installing git pre-push hook..."
	@cp scripts/pre-push .git/hooks/pre-push
	@chmod +x .git/hooks/pre-push
	@echo "✅ Pre-push hook installed!"
	@echo "   This will run 'make test-mlx-quick' before each push on Apple Silicon."

# Lint code
lint:
	@echo "Running linters..."
	-ruff check flash_attn/flash_attn_mlx/ --fix
	-ruff format flash_attn/flash_attn_mlx/

# =============================================================================
# Packaging
# =============================================================================

clean_dist:
	rm -rf dist/*

create_dist: clean_dist
	python setup.py sdist

upload_package: create_dist
	twine upload dist/*

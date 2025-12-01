# Contributing to Flash Attention

Thank you for your interest in contributing to Flash Attention!

## Development Setup

### Prerequisites

- Python 3.10+
- For MLX backend: Apple Silicon Mac (M1/M2/M3/M4)
- For CUDA backend: NVIDIA GPU with CUDA toolkit

### Installation

```bash
# Clone the repository
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention

# Install in development mode
make install-dev

# Or manually:
FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE pip install -e .
```

### Install Git Hooks (Recommended)

We provide a pre-push hook that runs quick MLX tests before pushing:

```bash
make install-hooks
```

This hook:
- Only runs on Apple Silicon Macs
- Only runs if MLX-related files were modified
- Runs a quick ~2 second smoke test
- Can be bypassed with `git push --no-verify`

## Testing

### Available Test Commands

```bash
# Show all available commands
make help

# Run all tests
make test

# MLX Backend (Apple Silicon only)
make test-mlx        # Run MLX test suite
make test-mlx-quick  # Quick smoke test (~5 seconds)
make test-mlx-full   # Full suite with coverage

# CUDA Backend
make test-cuda

# Benchmarks
make benchmark-mlx
```

### Before Submitting a PR

1. **Run tests locally:**
   ```bash
   # For MLX changes (Apple Silicon):
   make test-mlx
   
   # For CUDA changes:
   make test-cuda
   ```

2. **Run linters:**
   ```bash
   make lint
   ```

3. **Ensure CI passes:**
   - GitHub Actions will run syntax validation for MLX code
   - Full MLX tests require Apple Silicon (run locally)

## CI/CD Notes

### MLX Backend Testing

GitHub Actions runners are virtualized and don't have Metal GPU access. Therefore:

- **CI runs:** Python syntax validation only (no GPU execution)
- **Local testing required:** Full MLX tests must be run locally on Apple Silicon

The pre-push hook helps ensure MLX code works before pushing.

### Adding Tests

- MLX tests go in `flash_attn/flash_attn_mlx/tests/`
- Integration tests go in `tests/test_flash_attn_mlx.py`
- Use pytest fixtures from `conftest.py`

## Code Style

- Follow existing code patterns
- Use type hints
- Add docstrings for public functions
- Keep Metal kernels in `flash_attn/flash_attn_mlx/kernels/`

## Questions?

Open an issue or discussion on GitHub.

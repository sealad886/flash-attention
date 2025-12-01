"""Pytest configuration for Flash Attention MLX tests.

This file provides:
- Skip decorator for non-Apple Silicon platforms
- Common fixtures for MLX testing
- Automatic markers for test discovery
"""

from __future__ import annotations

import platform
import sys
from typing import Any, Dict, List

import pytest


def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon."""
    return sys.platform == "darwin" and platform.machine() == "arm64"


def mlx_available() -> bool:
    """Check if MLX is importable."""
    try:
        import mlx.core as mx
        return True
    except ImportError:
        return False


# Skip entire test file on non-Apple Silicon
skip_unless_apple_silicon = pytest.mark.skipif(
    not is_apple_silicon(),
    reason="MLX tests require Apple Silicon"
)

skip_unless_mlx = pytest.mark.skipif(
    not mlx_available(),
    reason="MLX package not installed"
)


@pytest.fixture
def mlx_seed() -> int:
    """Set a fixed seed for reproducible tests."""
    import mlx.core as mx
    mx.random.seed(42)
    return 42


@pytest.fixture
def tolerance() -> Dict[str, Dict[str, float]]:
    """Return tolerance dict for float comparison."""
    return {
        "float16": {"atol": 1e-2, "rtol": 1e-2},
        "float32": {"atol": 1e-4, "rtol": 1e-4},
        "bfloat16": {"atol": 2e-2, "rtol": 2e-2},
    }


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "mlx: MLX backend tests (Apple Silicon only)")
    config.addinivalue_line("markers", "cuda: CUDA backend tests (NVIDIA GPUs)")
    config.addinivalue_line("markers", "rocm: ROCm backend tests (AMD GPUs)")
    config.addinivalue_line("markers", "slow: Slow tests")


def pytest_collection_modifyitems(config: pytest.Config, items: List[pytest.Item]) -> None:
    """Automatically add mlx marker and skip non-Apple Silicon."""
    for item in items:
        # Check if test is in MLX test path
        if "flash_attn_mlx" in str(item.fspath) or "test_flash_attn_mlx" in str(item.fspath):
            item.add_marker(pytest.mark.mlx)

            # Skip on non-Apple Silicon
            if not is_apple_silicon():
                item.add_marker(pytest.mark.skip(reason="MLX tests require Apple Silicon"))
            elif not mlx_available():
                item.add_marker(pytest.mark.skip(reason="MLX package not installed"))

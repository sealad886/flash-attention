"""
MLX Device Detection and GPU Family Identification

This module provides utilities for detecting MLX availability and
identifying the Apple GPU family for performance tuning.
"""

import platform
import sys
from functools import lru_cache
from typing import Literal, Optional, Tuple


# GPU family type alias
GPUFamily = Literal["apple7", "apple8", "apple9", "unknown"]


@lru_cache(maxsize=1)
def is_apple_silicon() -> bool:
    """
    Check if running on Apple Silicon.

    Returns:
        True if running on macOS with ARM architecture (Apple Silicon)
    """
    return sys.platform == "darwin" and platform.machine() == "arm64"


@lru_cache(maxsize=1)
def is_mlx_available() -> bool:
    """
    Check if MLX framework is available.

    Returns:
        True if MLX is installed and can be imported
    """
    if not is_apple_silicon():
        return False

    try:
        import mlx.core as mx
        # Try a simple operation to ensure MLX is working
        _ = mx.array([1.0])
        return True
    except ImportError:
        return False
    except Exception:
        return False


@lru_cache(maxsize=1)
def get_gpu_family() -> GPUFamily:
    """
    Detect the Apple GPU family.

    Returns:
        GPU family string: "apple7" (M1), "apple8" (M2), "apple9" (M3+), or "unknown"

    Note:
        - Apple7: M1, M1 Pro, M1 Max, M1 Ultra
        - Apple8: M2, M2 Pro, M2 Max, M2 Ultra
        - Apple9: M3, M3 Pro, M3 Max, M4, M4 Pro, M4 Max
    """
    if not is_mlx_available():
        return "unknown"

    try:
        import mlx.core as mx

        # Get device info through MLX
        # MLX doesn't directly expose GPU family, so we use Metal API via subprocess
        # or infer from chip name
        chip_name = _get_chip_name()

        if chip_name is None:
            return "unknown"

        # Parse chip name to determine family
        chip_lower = chip_name.lower()

        if "m1" in chip_lower:
            return "apple7"
        elif "m2" in chip_lower:
            return "apple8"
        elif "m3" in chip_lower or "m4" in chip_lower:
            return "apple9"
        else:
            # Unknown chip, default to apple8 as middle ground
            return "unknown"

    except Exception:
        return "unknown"


def _get_chip_name() -> Optional[str]:
    """
    Get the Apple chip name using system_profiler.

    Returns:
        Chip name string (e.g., "Apple M1 Pro") or None if unavailable
    """
    import subprocess

    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass

    # Fallback: try system_profiler
    try:
        result = subprocess.run(
            ["system_profiler", "SPHardwareDataType"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "Chip:" in line or "Processor Name:" in line:
                    return line.split(":")[-1].strip()
    except Exception:
        pass

    return None


@lru_cache(maxsize=1)
def get_metal_capabilities() -> dict:
    """
    Get Metal GPU capabilities for tuning.

    Returns:
        Dictionary with capability information
    """
    if not is_mlx_available():
        return {"available": False}

    family = get_gpu_family()

    # Default capabilities based on GPU family
    capabilities = {
        "available": True,
        "gpu_family": family,
        "supports_bfloat16": family in ("apple8", "apple9"),  # BF16 support in M2+
        "max_threadgroup_memory": _get_threadgroup_memory_size(family),
        "simd_width": 32,  # Apple GPUs use SIMD32
    }

    return capabilities


def _get_threadgroup_memory_size(family: GPUFamily) -> int:
    """Get maximum threadgroup memory size in bytes."""
    if family == "apple7":
        return 32 * 1024  # 32 KB
    elif family == "apple8":
        return 32 * 1024  # 32 KB
    elif family == "apple9":
        return 64 * 1024  # 64 KB (M3+ has more)
    else:
        return 32 * 1024  # Default to conservative value


def get_block_sizes(headdim: int, gpu_family: Optional[GPUFamily] = None) -> Tuple[int, int]:
    """
    Get optimal BLOCK_M and BLOCK_N for given head dimension and GPU.

    Based on Metal Flash Attention tuning data.

    Args:
        headdim: Head dimension (32, 64, 96, 128, etc.)
        gpu_family: GPU family (auto-detected if None)

    Returns:
        Tuple of (BLOCK_M, BLOCK_N)
    """
    if gpu_family is None:
        gpu_family = get_gpu_family()

    # Block size lookup table based on Metal Flash Attention tuning
    # Format: (gpu_family, headdim) -> (BLOCK_M, BLOCK_N)
    block_sizes = {
        # Apple7 (M1)
        ("apple7", 32): (64, 64),
        ("apple7", 64): (64, 32),
        ("apple7", 96): (32, 32),
        ("apple7", 128): (32, 32),
        ("apple7", 160): (32, 32),
        ("apple7", 192): (32, 32),
        ("apple7", 256): (32, 16),

        # Apple8 (M2)
        ("apple8", 32): (64, 64),
        ("apple8", 64): (64, 64),
        ("apple8", 96): (32, 64),
        ("apple8", 128): (32, 32),
        ("apple8", 160): (32, 32),
        ("apple8", 192): (32, 32),
        ("apple8", 256): (32, 16),

        # Apple9 (M3+)
        ("apple9", 32): (32, 64),
        ("apple9", 64): (32, 64),
        ("apple9", 96): (32, 64),
        ("apple9", 128): (32, 32),
        ("apple9", 160): (32, 32),
        ("apple9", 192): (32, 32),
        ("apple9", 256): (32, 16),
    }

    key = (gpu_family, headdim)
    if key in block_sizes:
        return block_sizes[key]

    # Fallback: find closest headdim
    for hd in [32, 64, 96, 128, 160, 192, 256]:
        if hd >= headdim:
            fallback_key = (gpu_family, hd)
            if fallback_key in block_sizes:
                return block_sizes[fallback_key]

    # Ultimate fallback
    return (32, 32)


def check_mlx_requirements() -> Tuple[bool, str]:
    """
    Check if all requirements for MLX Flash Attention are met.

    Returns:
        Tuple of (success, message)
    """
    if not is_apple_silicon():
        return False, "MLX Flash Attention requires Apple Silicon (M1/M2/M3)"

    if not is_mlx_available():
        return False, "MLX not installed. Install with: pip install mlx"

    try:
        import mlx.core as mx

        # Check MLX version
        # MLX 0.20.0+ required for metal_kernel API
        version = getattr(mx, "__version__", "0.0.0")
        major, minor, patch = map(int, version.split(".")[:3])

        if major == 0 and minor < 20:
            return False, f"MLX version {version} is too old. Requires >= 0.20.0"

    except Exception as e:
        return False, f"Error checking MLX: {e}"

    return True, f"MLX Flash Attention ready (GPU: {get_gpu_family()})"


def get_device_info() -> dict:
    """
    Get comprehensive device information.

    Returns:
        Dictionary with device information
    """
    info = {
        "platform": sys.platform,
        "machine": platform.machine(),
        "is_apple_silicon": is_apple_silicon(),
        "mlx_available": is_mlx_available(),
    }

    if is_mlx_available():
        import mlx.core as mx
        info.update({
            "mlx_version": getattr(mx, "__version__", "unknown"),
            "gpu_family": get_gpu_family(),
            "chip_name": _get_chip_name(),
            "capabilities": get_metal_capabilities(),
        })

    return info

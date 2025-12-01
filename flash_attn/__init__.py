__version__ = "2.8.3"

import platform
import sys

def _is_apple_silicon():
    """Check if running on Apple Silicon."""
    return sys.platform == "darwin" and platform.machine() == "arm64"


# Platform-aware imports
if _is_apple_silicon():
    try:
        # On Apple Silicon, use MLX backend
        from flash_attn.flash_attn_mlx import (
            flash_attn_func,
            flash_attn_kvpacked_func,
            flash_attn_qkvpacked_func,
            flash_attn_varlen_func,
            flash_attn_varlen_kvpacked_func,
            flash_attn_varlen_qkvpacked_func,
            flash_attn_with_kvcache,
        )
    except ImportError as e:
        raise ImportError(
            "MLX backend required on Apple Silicon. Install with: pip install mlx"
        ) from e
else:
    # On other platforms, use CUDA/ROCm backend
    from flash_attn.flash_attn_interface import (
        flash_attn_func,
        flash_attn_kvpacked_func,
        flash_attn_qkvpacked_func,
        flash_attn_varlen_func,
        flash_attn_varlen_kvpacked_func,
        flash_attn_varlen_qkvpacked_func,
        flash_attn_with_kvcache,
    )

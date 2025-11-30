"""
Flash Attention Forward Kernel Wrapper

This module provides Python wrappers for invoking Flash Attention Metal kernels
using MLX's mx.fast.metal_kernel API.
"""

import math
from typing import Callable, Optional, Tuple, cast

import mlx.core as mx

from .device import get_block_sizes, get_gpu_family
from .kernels import get_attention_forward_kernel
from .params import AttentionParams, create_attention_params
from .utils import log2_if_power_of_two


# ============================================================================
# Kernel Factory and Wrapper
# ============================================================================

_attention_fwd_kernel = None
_DUMMY_PAGE_TABLE = mx.zeros((1,), dtype=mx.int32)


def _get_attention_fwd_kernel():
    """Get or create the attention forward kernel."""
    global _attention_fwd_kernel

    if _attention_fwd_kernel is None:
        source = get_attention_forward_kernel()
        _attention_fwd_kernel = mx.fast.metal_kernel(
            name="flash_attention_fwd",
            input_names=["Q", "K", "V", "batch_size", "seqlen_q", "seqlen_k",
                        "nheads", "nheads_k", "headdim", "scale", "causal",
                        "softcap", "window_left", "window_right",
                        "alibi_slopes", "use_alibi", "alibi_batch_stride",
                        "page_table", "page_size", "max_pages_per_seq", "use_paged_kv",
                        "log2_page_size",
                        "dropout_p", "philox_seed", "philox_offset"],
            output_names=["O", "L"],
            source=source,
            ensure_row_contiguous=True,
        )

    return _attention_fwd_kernel


def flash_attention_forward_metal(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    softcap: float = 0.0,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[mx.array] = None,
    dropout_p: float = 0.0,
    philox_seed: int = 0,
    philox_offset: int = 0,
    page_table: Optional[mx.array] = None,
    page_size: Optional[int] = None,
    max_pages_per_seq: Optional[int] = None,
) -> Tuple[mx.array, mx.array]:
    """
    Flash Attention forward pass using Metal kernel.

    Args:
        q: Query tensor of shape (batch, seqlen_q, nheads, headdim)
          k: Key tensor of shape (batch, seqlen_k, nheads_k, headdim) or
              (num_pages, page_size, nheads_k, headdim) when block_table is provided
          v: Value tensor with the same layout as ``k``
        softmax_scale: Scaling factor (default: 1/sqrt(headdim))
        causal: Whether to apply causal masking
        softcap: Soft cap for attention logits (0 = disabled)
        window_size: (left, right) window sizes for sliding window attention
                     (-1, -1) means full attention
        alibi_slopes: ALiBi slopes of shape (nheads,) or (batch, nheads)
        dropout_p: Dropout probability (0 = no dropout)
        philox_seed: RNG seed for dropout (for reproducibility)
        philox_offset: RNG offset for dropout
        page_table: Optional page/block table for paged KV caches
        page_size: Logical page size when paged KV is enabled
        max_pages_per_seq: Maximum number of pages per sequence when paged KV is enabled

    Returns:
        Tuple of:
        - output: Attention output of shape (batch, seqlen_q, nheads, headdim)
        - softmax_lse: Log-sum-exp of shape (batch, nheads, seqlen_q)
    """
    batch, seqlen_q, nheads, headdim = q.shape
    _, default_seqlen_k, nheads_k, _ = k.shape

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(headdim)

    paged_kv_enabled = page_table is not None
    if paged_kv_enabled:
        if page_size is None or max_pages_per_seq is None:
            raise ValueError("page_size and max_pages_per_seq must be provided for paged KV")
        page_size_value = int(page_size)
        max_pages_value = int(max_pages_per_seq)
        if page_size_value <= 0 or max_pages_value <= 0:
            raise ValueError("page_size and max_pages_per_seq must be positive integers")
        seqlen_k = page_size_value * max_pages_value
        page_table_input = mx.array(page_table, dtype=mx.int32)
        log2_page_size = log2_if_power_of_two(page_size_value)
    else:
        seqlen_k = default_seqlen_k
        page_size_value = 1
        max_pages_value = 1
        log2_page_size = -1
        page_table_input = _DUMMY_PAGE_TABLE

    # Get kernel
    kernel: Callable = cast(Callable, _get_attention_fwd_kernel())

    # Calculate grid size: one thread per (batch, query_pos, head) tuple
    total_elements = batch * seqlen_q * nheads

    # Choose threadgroup size (multiple of 32 for SIMD efficiency)
    threadgroup_size = 256

    # Prepare scalar inputs
    causal_int = 1 if causal else 0
    window_left, window_right = window_size

    # Handle ALiBi parameters
    if alibi_slopes is not None:
        use_alibi = 1
        alibi_slopes = mx.array(alibi_slopes, dtype=mx.float32)
        if alibi_slopes.ndim == 1:
            alibi_batch_stride = 0  # (nheads,) shape
        else:
            alibi_batch_stride = alibi_slopes.shape[1]  # (batch, nheads) shape
    else:
        use_alibi = 0
        alibi_slopes = mx.zeros((1,), dtype=mx.float32)  # Dummy array
        alibi_batch_stride = 0

    # Output shapes
    output_shape = (batch, seqlen_q, nheads, headdim)
    lse_shape = (batch, nheads, seqlen_q)

    # Invoke kernel with dropout parameters
    outputs = kernel(
        inputs=[q, k, v, batch, seqlen_q, seqlen_k,
                nheads, nheads_k, headdim, softmax_scale, causal_int, softcap,
                window_left, window_right,
                alibi_slopes, use_alibi, alibi_batch_stride,
                page_table_input, page_size_value, max_pages_value, 1 if paged_kv_enabled else 0,
                log2_page_size,
                dropout_p, philox_seed, philox_offset],
        template=[("T", q.dtype)],
        grid=(total_elements, 1, 1),
        threadgroup=(threadgroup_size, 1, 1),
        output_shapes=[output_shape, lse_shape],
        output_dtypes=[q.dtype, mx.float32],
    )

    return outputs[0], outputs[1]


# ============================================================================
# Alternative: Use Reference Implementation
# ============================================================================

def flash_attention_forward_reference(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    softcap: float = 0.0,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[mx.array] = None,
    dropout_p: float = 0.0,
    philox_seed: int = 0,
    philox_offset: int = 0,
) -> Tuple[mx.array, mx.array]:
    """
    Flash Attention forward using reference implementation.

    Falls back to this when Metal kernel is not suitable or for validation.
    """
    from .reference import attention_ref_mlx

    out, lse, _ = attention_ref_mlx(
        q, k, v,
        softmax_scale=softmax_scale,
        causal=causal,
        softcap=softcap,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        dropout_p=dropout_p,
    )
    return out, lse


# ============================================================================
# Main Forward Function
# ============================================================================

def flash_attention_forward(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    softcap: float = 0.0,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[mx.array] = None,
    dropout_p: float = 0.0,
    philox_seed: int = 0,
    philox_offset: int = 0,
    use_metal_kernel: bool = True,
    block_table: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    """
    Flash Attention forward pass.

    Automatically selects between Metal kernel and reference implementation
    based on feature requirements and configuration.

    Args:
        q: Query tensor of shape (batch, seqlen_q, nheads, headdim)
        k: Key tensor of shape (batch, seqlen_k, nheads_k, headdim)
        v: Value tensor of shape (batch, seqlen_k, nheads_k, headdim)
        softmax_scale: Scaling factor (default: 1/sqrt(headdim))
        causal: Whether to apply causal masking
        softcap: Soft cap for attention logits (0 = disabled)
        window_size: (left, right) window sizes for local attention
        alibi_slopes: ALiBi slopes for positional encoding
        dropout_p: Dropout probability (0 = no dropout)
        philox_seed: RNG seed for dropout
        philox_offset: RNG offset for dropout
        use_metal_kernel: Whether to try using Metal kernel
        block_table: Optional page/block table (batch, max_pages_per_seq) enabling paged KV caches

    Returns:
        Tuple of:
        - output: Attention output of shape (batch, seqlen_q, nheads, headdim)
        - softmax_lse: Log-sum-exp of shape (batch, nheads, seqlen_q)
    """
    batch, seqlen_q, nheads, headdim = q.shape
    _, _, nheads_k, _ = k.shape

    paged_kv_enabled = block_table is not None
    page_table: Optional[mx.array] = None
    page_size: Optional[int] = None
    max_pages_per_seq: Optional[int] = None

    if paged_kv_enabled:
        page_table = mx.array(block_table, dtype=mx.int32)
        if page_table.ndim != 2:
            raise ValueError(f"block_table must be 2D, got {page_table.ndim}D")
        if page_table.shape[0] != batch:
            raise ValueError(
                f"block_table batch dimension ({page_table.shape[0]}) must match query batch ({batch})",
            )
        page_size = int(k.shape[1])
        max_pages_per_seq = int(page_table.shape[1])
    else:
        page_size = None
        max_pages_per_seq = None

    # Metal kernel now supports ALiBi and dropout
    if not use_metal_kernel:
        return flash_attention_forward_reference(
            q, k, v,
            softmax_scale=softmax_scale,
            causal=causal,
            softcap=softcap,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            dropout_p=dropout_p,
            philox_seed=philox_seed,
            philox_offset=philox_offset,
        )

    # Use Metal kernel (supports causal, softcap, sliding window, ALiBi, and dropout)
    try:
        return flash_attention_forward_metal(
            q, k, v,
            softmax_scale=softmax_scale,
            causal=causal,
            softcap=softcap,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            dropout_p=dropout_p,
            philox_seed=philox_seed,
            philox_offset=philox_offset,
            page_table=page_table,
            page_size=page_size,
            max_pages_per_seq=max_pages_per_seq,
        )
    except Exception:
        if paged_kv_enabled:
            raise
        # Fall back to reference on any kernel error (contiguous caches only)
        return flash_attention_forward_reference(
            q, k, v,
            softmax_scale=softmax_scale,
            causal=causal,
            softcap=softcap,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            dropout_p=dropout_p,
            philox_seed=philox_seed,
            philox_offset=philox_offset,
        )

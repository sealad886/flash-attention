"""
Flash Attention Backward Kernels for MLX

This module implements the backward pass for Flash Attention using MLX Metal kernels.
Uses the split backward design (separate dQ and dK/dV kernels) to avoid FP32 atomics.

Based on the Metal Flash Attention paper's alternative algorithm:
- dQ kernel: parallelized over query positions
- dK/dV kernel: parallelized over key/value positions
"""

import math
from typing import Callable, Tuple, Union, cast

import mlx.core as mx

from .kernels import get_attention_backward_dq_kernel, get_attention_backward_dkv_kernel
from .reference import attention_backward_ref


# Cached kernel instances
_dq_kernel = None
_dkv_kernel = None


def _get_dq_kernel():
    """Get or create the dQ backward kernel."""
    global _dq_kernel
    if _dq_kernel is None:
        source = get_attention_backward_dq_kernel()
        _dq_kernel = mx.fast.metal_kernel(
            name="flash_attention_backward_dq",
            input_names=["Q", "K", "V", "O", "dO", "L",
                        "batch_size", "seq_len_q", "seq_len_k",
                        "num_heads", "num_heads_k", "head_dim",
                        "softmax_scale", "causal_flag",
                        "window_left", "window_right",
                        "alibi_slopes", "use_alibi", "alibi_batch_stride",
                        "dropout_p", "philox_seed", "philox_offset"],
            output_names=["dQ"],
            source=source,
            header="""
                #include <metal_stdlib>
                using namespace metal;
            """,
            ensure_row_contiguous=True,
        )
    return _dq_kernel


def _get_dkv_kernel():
    """Get or create the dK/dV backward kernel."""
    global _dkv_kernel
    if _dkv_kernel is None:
        source = get_attention_backward_dkv_kernel()
        _dkv_kernel = mx.fast.metal_kernel(
            name="flash_attention_backward_dkv",
            input_names=["Q", "K", "V", "O", "dO", "L",
                        "batch_size", "seq_len_q", "seq_len_k",
                        "num_heads", "num_heads_k", "head_dim",
                        "softmax_scale", "causal_flag",
                        "window_left", "window_right",
                        "alibi_slopes", "use_alibi", "alibi_batch_stride",
                        "dropout_p", "philox_seed", "philox_offset"],
            output_names=["dK", "dV"],
            source=source,
            header="""
                #include <metal_stdlib>
                using namespace metal;
            """,
            ensure_row_contiguous=True,
        )
    return _dkv_kernel


def flash_attention_backward(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    o: mx.array,
    dout: mx.array,
    softmax_lse: mx.array,
    softmax_scale: float = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: mx.array = None,
    dropout_p: float = 0.0,
    philox_seed: int = 0,
    philox_offset: int = 0,
    use_metal_kernel: bool = True,
) -> Tuple[mx.array, mx.array, mx.array]:
    """
    Compute backward pass for Flash Attention.

    Uses split backward design with separate dQ and dK/dV kernels.

    Args:
        q: Query tensor [batch, seqlen_q, nheads, headdim]
        k: Key tensor [batch, seqlen_k, nheads_k, headdim]
        v: Value tensor [batch, seqlen_k, nheads_k, headdim]
        o: Forward output [batch, seqlen_q, nheads, headdim]
        dout: Gradient of output [batch, seqlen_q, nheads, headdim]
        softmax_lse: Logsumexp from forward [batch, nheads, seqlen_q]
        softmax_scale: Scaling factor (default 1/sqrt(headdim))
        causal: Whether to apply causal masking
        window_size: (left, right) for sliding window attention
        alibi_slopes: ALiBi slopes of shape (nheads,) or (batch, nheads)
        dropout_p: Dropout probability (0 = no dropout)
        philox_seed: RNG seed for dropout mask regeneration
        philox_offset: RNG offset for dropout mask regeneration
        use_metal_kernel: Whether to use Metal kernel (True) or reference (False)

    Returns:
        dq: Gradient for queries
        dk: Gradient for keys
        dv: Gradient for values
    """
    batch, seqlen_q, nheads, headdim = q.shape
    _, seqlen_k, nheads_k, _ = k.shape

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(headdim)

    # Fall back to reference if requested
    if not use_metal_kernel:
        return attention_backward_ref(
            q, k, v, o, dout, softmax_lse,
            softmax_scale=softmax_scale,
            causal=causal,
        )

    dtype = q.dtype
    causal_int = 1 if causal else 0
    window_left, window_right = window_size

    # Handle ALiBi parameters
    if alibi_slopes is not None:
        use_alibi = 1
        alibi_slopes_arr = mx.array(alibi_slopes, dtype=mx.float32)
        if alibi_slopes_arr.ndim == 1:
            alibi_batch_stride = 0  # (nheads,) shape
        else:
            alibi_batch_stride = alibi_slopes_arr.shape[1]  # (batch, nheads) shape
    else:
        use_alibi = 0
        alibi_slopes_arr = mx.zeros((1,), dtype=mx.float32)  # Dummy array
        alibi_batch_stride = 0

    try:
        # Get kernels (cached)
        dq_kernel: Callable = cast(Callable, _get_dq_kernel())
        dkv_kernel: Callable = cast(Callable, _get_dkv_kernel())

        # Run dQ kernel
        # Grid: (batch * nheads, seqlen_q, 1)
        dq_grid = (batch * nheads, seqlen_q, 1)
        dq_threadgroup = (1, 1, 1)

        dq_outputs = cast(Callable, dq_kernel(
            inputs=[q, k, v, o, dout, softmax_lse,
                   batch, seqlen_q, seqlen_k,
                   nheads, nheads_k, headdim,
                   softmax_scale, causal_int,
                   window_left, window_right,
                   alibi_slopes_arr, use_alibi, alibi_batch_stride,
                   dropout_p, philox_seed, philox_offset],
            template=[("T", dtype)],
            output_shapes=[q.shape],
            output_dtypes=[dtype],
            grid=dq_grid,
            threadgroup=dq_threadgroup,
            init_value=0.0,
        ))
        dq = dq_outputs[0]  # type: ignore

        # Run dK/dV kernel
        # Grid: (batch * nheads_k, seqlen_k, 1)
        dkv_grid = (batch * nheads_k, seqlen_k, 1)
        dkv_threadgroup = (1, 1, 1)

        dkv_outputs = dkv_kernel(
            inputs=[q, k, v, o, dout, softmax_lse,
                   batch, seqlen_q, seqlen_k,
                   nheads, nheads_k, headdim,
                   softmax_scale, causal_int,
                   window_left, window_right,
                   alibi_slopes_arr, use_alibi, alibi_batch_stride,
                   dropout_p, philox_seed, philox_offset],
            template=[("T", dtype)],
            output_shapes=[k.shape, v.shape],
            output_dtypes=[dtype, dtype],
            grid=dkv_grid,
            threadgroup=dkv_threadgroup,
            init_value=0.0,
        )
        dk, dv = dkv_outputs

        return dq, dk, dv

    except Exception:
        # Fall back to reference implementation
        return attention_backward_ref(
            q, k, v, o, dout, softmax_lse,
            softmax_scale=softmax_scale,
            causal=causal,
        )

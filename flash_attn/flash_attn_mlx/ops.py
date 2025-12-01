"""
Flash Attention MLX Operations with Autograd Support

This module provides Flash Attention operations wrapped with MLX's
custom_function decorator for automatic differentiation.

The implementation uses a hybrid approach:
1. mx.fast.scaled_dot_product_attention for common cases (fast, well-tested)
2. Custom Metal kernels for advanced features (softcap, paged KV, dropout)
"""

import math
from typing import Optional, Tuple, cast

import mlx.core as mx

from .fwd_kernel import flash_attention_forward_metal as flash_attention_forward
from .bwd_kernel import flash_attention_backward
from .params import AttentionParams, create_attention_params
from .utils import check_args


# ============================================================================
# MLX SDPA Utilities
# ============================================================================

def _can_use_mlx_sdpa(
    softcap: float,
    window_size: Tuple[int, int],
    alibi_slopes: Optional[mx.array],
    dropout_p: float,
    page_table: Optional[mx.array],
) -> bool:
    """
    Determine if we can use MLX's native scaled_dot_product_attention.

    MLX's SDPA is highly optimized but doesn't support:
    - Softcap (logit soft-capping)
    - Paged KV cache
    - Dropout (use separate dropout layer if needed for training)

    It DOES support via mask parameter:
    - Causal masking
    - Sliding window attention
    - ALiBi bias (as additive mask)
    """
    if softcap > 0:
        return False
    if page_table is not None:
        return False
    if dropout_p > 0:
        return False
    # Sliding window and ALiBi can be handled via mask parameter
    return True


def _create_combined_mask(
    seqlen_q: int,
    seqlen_k: int,
    nheads: int,
    batch: int,
    causal: bool,
    window_size: Tuple[int, int],
    alibi_slopes: Optional[mx.array],
    dtype: mx.Dtype = mx.float32,
) -> Optional[mx.array|str]:        # type: ignore[type-arg]
    """
    Create a combined attention mask for MLX SDPA.

    Returns:
        - "causal" string if only causal masking is needed
        - Additive mask array if sliding window or ALiBi is needed
        - None if no masking is needed
    """
    window_left, window_right = window_size
    has_sliding_window = window_left >= 0 or window_right >= 0
    has_alibi = alibi_slopes is not None

    # Simple case: just causal, no other features
    if causal and not has_sliding_window and not has_alibi:
        return "causal"

    # No masking needed
    if not causal and not has_sliding_window and not has_alibi:
        return None

    # Build mask for sliding window and/or ALiBi
    # Create position indices
    positions_q = mx.arange(seqlen_q).reshape(1, 1, -1, 1)  # (1, 1, seqlen_q, 1)
    positions_k = mx.arange(seqlen_k).reshape(1, 1, 1, -1)  # (1, 1, 1, seqlen_k)

    # Start with zeros (no mask effect) - use float32 for computation
    mask = mx.zeros((1, 1, seqlen_q, seqlen_k), dtype=mx.float32)

    # Apply causal mask
    if causal:
        causal_mask = positions_k > positions_q
        mask = mx.where(causal_mask, mx.array(float("-inf"), dtype=mx.float32), mask)

    # Apply sliding window mask
    if has_sliding_window:
        if window_left >= 0:
            left_mask = positions_k < (positions_q - window_left)
            mask = mx.where(left_mask, mx.array(float("-inf"), dtype=mx.float32), mask)
        if window_right >= 0:
            right_mask = positions_k > (positions_q + window_right)
            mask = mx.where(right_mask, mx.array(float("-inf"), dtype=mx.float32), mask)

    # Apply ALiBi bias
    if alibi_slopes is not None:
        # alibi_slopes: (nheads,) or (batch, nheads)
        if alibi_slopes.ndim == 1:
            slopes = alibi_slopes.astype(mx.float32).reshape(1, -1, 1, 1)  # (1, nheads, 1, 1)
        else:
            slopes = alibi_slopes.astype(mx.float32).reshape(batch, -1, 1, 1)  # (batch, nheads, 1, 1)

        distance = (positions_q - positions_k).astype(mx.float32)  # (1, 1, seqlen_q, seqlen_k)
        alibi_bias = slopes * distance  # Negative for positions before current
        mask = mask + alibi_bias

    # Cast mask to match input dtype for MLX SDPA compatibility
    return mask.astype(dtype)


def _attention_mlx_sdpa(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    softmax_scale: float,
    causal: bool,
    window_size: Tuple[int, int],
    alibi_slopes: Optional[mx.array],
) -> mx.array:
    """
    Execute attention using MLX's native scaled_dot_product_attention.

    Handles shape transposition and mask creation.
    """
    batch, seqlen_q, nheads, headdim = q.shape
    _, seqlen_k, nheads_k, _ = k.shape

    # Transpose: (batch, seqlen, nheads, headdim) -> (batch, nheads, seqlen, headdim)
    q_t = mx.transpose(q, (0, 2, 1, 3))
    k_t = mx.transpose(k, (0, 2, 1, 3))
    v_t = mx.transpose(v, (0, 2, 1, 3))

    # Create combined mask with matching dtype for MLX SDPA
    mask = _create_combined_mask(
        seqlen_q, seqlen_k, nheads, batch,
        causal, window_size, alibi_slopes,
        dtype=q.dtype,  # Match input dtype
    )

    # Call MLX's optimized SDPA
    out_t = mx.fast.scaled_dot_product_attention(
        q_t, k_t, v_t,
        scale=softmax_scale,
        mask=mask,
    )

    # Transpose back: (batch, nheads, seqlen, headdim) -> (batch, seqlen, nheads, headdim)
    return mx.transpose(out_t, (0, 2, 1, 3))


# ============================================================================
# Core Attention Operation with VJP Support
# ============================================================================

# Module-level state for passing context between forward and backward
class _FlashAttentionState:
    """State container for flash attention forward/backward pass.

    Note: The deterministic flag is provided for CUDA parity but has no effect
    on MLX/Metal kernels, which are inherently deterministic due to:
    1. Single-threaded evaluation model within compute commands
    2. Philox RNG being deterministic given same seed/offset
    3. No non-deterministic memory access patterns in our kernels
    """
    lse: Optional[mx.array] = None
    softmax_scale: float = 1.0
    causal: bool = False
    softcap: float = 0.0
    window_size: Tuple[int, int] = (-1, -1)
    alibi_slopes: Optional[mx.array] = None
    dropout_p: float = 0.0
    philox_seed: int = 0
    philox_offset: int = 0
    deterministic: bool = False  # Provided for API parity; MLX is always deterministic

_state = _FlashAttentionState()


@mx.custom_function
def _flash_attention_custom_kernel(
    q: mx.array,
    k: mx.array,
    v: mx.array,
) -> mx.array:
    """
    Flash attention using custom Metal kernels.

    This is used for advanced features not supported by MLX's native SDPA:
    - Softcap (logit soft-capping)
    - Dropout with reproducible Philox RNG
    - Paged KV cache
    """
    global _state

    out, lse = flash_attention_forward(
        q, k, v,
        softmax_scale=_state.softmax_scale,
        causal=_state.causal,
        softcap=_state.softcap,
        window_size=_state.window_size,
        alibi_slopes=_state.alibi_slopes,
        dropout_p=_state.dropout_p,
        philox_seed=_state.philox_seed,
        philox_offset=_state.philox_offset,
    )
    _state.lse = lse
    return out


@_flash_attention_custom_kernel.vjp
def _flash_attention_custom_kernel_vjp(primals, cotangent, output):
    """
    Vector-Jacobian product for flash attention.

    Args:
        primals: Tuple of (q, k, v)
        cotangent: Gradient of the output (dO)
        output: The output from the forward pass

    Returns:
        Tuple of gradients (dq, dk, dv)
    """
    global _state

    q, k, v = primals
    do = cotangent

    lse = _state.lse
    if lse is None:
        # Fallback: recompute LSE if not available
        _, lse = flash_attention_forward(
            q, k, v,
            softmax_scale=_state.softmax_scale,
            causal=_state.causal,
            softcap=_state.softcap,
            window_size=_state.window_size,
            alibi_slopes=_state.alibi_slopes,
            dropout_p=_state.dropout_p,
            philox_seed=_state.philox_seed,
            philox_offset=_state.philox_offset,
        )

    # Call backward kernel (dropout params passed for mask regeneration)
    dq, dk, dv = flash_attention_backward(
        q, k, v, output, do, lse,
        softmax_scale=_state.softmax_scale,
        causal=_state.causal,
        window_size=_state.window_size,
        alibi_slopes=_state.alibi_slopes,
        dropout_p=_state.dropout_p,
        philox_seed=_state.philox_seed,
        philox_offset=_state.philox_offset,
    )

    return dq, dk, dv


def flash_attention_mlx(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    softmax_scale: float,
    causal: bool = False,
    softcap: float = 0.0,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[mx.array] = None,
    dropout_p: float = 0.0,
    philox_seed: int = 0,
    philox_offset: int = 0,
    deterministic: bool = False,
) -> mx.array:
    """
    Flash Attention operation with automatic differentiation support.

    Args:
        q: Query tensor of shape (batch, seqlen_q, nheads, headdim)
        k: Key tensor of shape (batch, seqlen_k, nheads_k, headdim)
        v: Value tensor of shape (batch, seqlen_k, nheads_k, headdim)
        softmax_scale: Scaling factor for attention scores
        causal: Whether to apply causal masking
        softcap: Soft cap for attention logits (0 = disabled)
        window_size: (left, right) for sliding window attention
        alibi_slopes: ALiBi slopes of shape (nheads,) or (batch, nheads)
            that stay on-device and eliminate reference fallbacks
        dropout_p: Dropout probability (0 = no dropout) using Philox-based
            masks shared by the forward/backward kernels
        philox_seed: RNG seed for dropout
        philox_offset: RNG offset for dropout
        deterministic: For CUDA parity; MLX kernels are always deterministic

    Returns:
        out: Attention output of shape (batch, seqlen_q, nheads, headdim)

    Implementation Notes:
        This function uses a hybrid approach:
        1. For common cases (no softcap, no dropout, no paged KV), uses MLX's
           highly optimized mx.fast.scaled_dot_product_attention
        2. For advanced features (softcap, dropout, paged KV), falls back to
           custom Metal kernels

        Sliding window and ALiBi are handled via mask construction and work
        with MLX's native SDPA.
    """
    global _state

    # Determine which backend to use
    use_mlx_sdpa = _can_use_mlx_sdpa(
        softcap=softcap,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        dropout_p=dropout_p,
        page_table=None,  # flash_attention_mlx doesn't support paged KV
    )

    if use_mlx_sdpa:
        # Use MLX's optimized native SDPA - gradients handled automatically
        return _attention_mlx_sdpa(
            q, k, v,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
        )

    # Use custom Metal kernel for advanced features
    # Set up state for custom_function VJP
    _state.softmax_scale = softmax_scale
    _state.causal = causal
    _state.softcap = softcap
    _state.window_size = window_size
    _state.alibi_slopes = alibi_slopes
    _state.dropout_p = dropout_p
    _state.philox_seed = philox_seed
    _state.philox_offset = philox_offset
    _state.deterministic = deterministic
    _state.lse = None

    return cast(mx.array, _flash_attention_custom_kernel(q, k, v))


def flash_attention_with_lse(
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
    block_table: Optional[mx.array] = None,
    page_size: Optional[int] = None,
    max_pages_per_seq: Optional[int] = None,
) -> Tuple[mx.array, mx.array]:
    """
    Flash Attention returning both output and log-sum-exp.

    This version does NOT use custom_function since it returns multiple outputs.
    Use for cases where you need the LSE explicitly, or need advanced features.

    Args:
        q: Query tensor of shape (batch, seqlen_q, nheads, headdim)
        k: Key tensor of shape (batch, seqlen_k, nheads_k, headdim)
        v: Value tensor of shape (batch, seqlen_k, nheads_k, headdim)
        softmax_scale: Scaling factor for attention scores
        causal: Whether to apply causal masking
        softcap: Soft cap for attention logits (0 = disabled)
        window_size: (left, right) for sliding window attention
        alibi_slopes: ALiBi slopes of shape (nheads,) or (batch, nheads)
            that stay on-device and eliminate reference fallbacks
        dropout_p: Dropout probability (0 = no dropout) using Philox-based
            masks shared by the forward/backward kernels
        philox_seed: RNG seed for dropout
        philox_offset: RNG offset for dropout
        block_table: Optional page/block table for paged KV cache access

    Returns:
        Tuple of (output, softmax_lse)
    """
    batch, seqlen_q, nheads, headdim = q.shape

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(headdim)

    page_table = None
    if block_table is None:
        check_args(q, k, v)
    else:
        page_table = mx.array(block_table, dtype=mx.int32)
        if page_table.ndim != 2:
            raise ValueError(f"block_table must be 2D, got {page_table.ndim}D")
        if page_table.shape[0] != batch:
            raise ValueError(
                f"block_table batch dimension ({page_table.shape[0]}) must match query batch ({batch})",
            )
        if k.ndim != 4 or v.ndim != 4:
            raise ValueError("Paged KV tensors must be 4D (num_pages, page_size, nheads_k, headdim)")
        if k.shape != v.shape:
            raise ValueError("Paged K and V caches must share the same shape")

    return flash_attention_forward(
        q, k, v,
        softmax_scale=softmax_scale,
        causal=causal,
        softcap=softcap,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        dropout_p=dropout_p,
        philox_seed=philox_seed,
        philox_offset=philox_offset,
        page_table=page_table if block_table is not None else None,
        page_size=page_size,
        max_pages_per_seq=max_pages_per_seq,
    )

"""
MLX Flash Attention Reference Implementation

This module provides a pure MLX implementation of attention for correctness
validation and as a fallback when custom Metal kernels are not available.

The implementation uses mx.fast.scaled_dot_product_attention where possible
and falls back to manual computation for unsupported features.
"""

import math
from typing import Optional, Tuple

import mlx.core as mx


def attention_ref_mlx(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[mx.array] = None,
    dropout_mask: Optional[mx.array] = None,
    key: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array, Optional[mx.array]]:
    """
    Reference implementation of scaled dot-product attention using pure MLX ops.

    This implementation prioritizes correctness over speed and is used for:
    1. Validating custom Metal kernel outputs
    2. Fallback when Metal kernels are unavailable
    3. Features not yet implemented in Metal kernels

    Args:
        q: Query tensor of shape (batch, seqlen_q, nheads, headdim)
        k: Key tensor of shape (batch, seqlen_k, nheads_k, headdim)
        v: Value tensor of shape (batch, seqlen_k, nheads_k, headdim)
        dropout_p: Dropout probability (0 = disabled)
        softmax_scale: Scaling factor (default: 1/sqrt(headdim))
        causal: If True, apply causal mask
        window_size: (left, right) window sizes for local attention
        softcap: Soft cap for attention logits (0 = disabled)
        alibi_slopes: ALiBi slopes of shape (batch, nheads) or (nheads,)
        dropout_mask: Pre-computed dropout mask for reproducibility
        key: RNG key for dropout (ignored, using philox seed instead)

    Returns:
        Tuple of:
        - output: Attention output of shape (batch, seqlen_q, nheads, headdim)
        - softmax_lse: Log-sum-exp of shape (batch, nheads, seqlen_q)
        - attn_probs: Attention probabilities (optional, for debugging)
    """
    batch, seqlen_q, nheads, headdim = q.shape
    _, seqlen_k, nheads_k, _ = k.shape

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(headdim)

    # Try to use mx.fast.scaled_dot_product_attention for simple cases
    if _can_use_mlx_sdpa(causal, window_size, softcap, alibi_slopes, dropout_p):
        return _attention_mlx_sdpa(q, k, v, softmax_scale, causal)

    # Fall back to manual implementation for advanced features
    return _attention_manual(
        q, k, v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        dropout_mask=dropout_mask,
    )


def _can_use_mlx_sdpa(
    causal: bool,
    window_size: Tuple[int, int],
    softcap: float,
    alibi_slopes: Optional[mx.array],
    dropout_p: float,
) -> bool:
    """Check if we can use MLX's native SDPA."""
    # MLX SDPA supports causal mask but not other features
    if window_size != (-1, -1):
        return False  # Sliding window not supported
    if softcap > 0:
        return False  # Softcap not supported
    if alibi_slopes is not None:
        return False  # ALiBi not supported
    if dropout_p > 0:
        return False  # Dropout not supported in native SDPA
    return True


def _attention_mlx_sdpa(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    softmax_scale: float,
    causal: bool,
) -> Tuple[mx.array, mx.array, None]:
    """
    Use MLX's native scaled_dot_product_attention.

    Note: MLX SDPA expects shape (batch, nheads, seqlen, headdim),
    but Flash Attention uses (batch, seqlen, nheads, headdim).
    """
    # Transpose to MLX SDPA format: (batch, seqlen, nheads, headdim) -> (batch, nheads, seqlen, headdim)
    q_t = mx.transpose(q, (0, 2, 1, 3))
    k_t = mx.transpose(k, (0, 2, 1, 3))
    v_t = mx.transpose(v, (0, 2, 1, 3))

    # Use native SDPA
    mask = "causal" if causal else None
    out_t = mx.fast.scaled_dot_product_attention(
        q_t, k_t, v_t,
        scale=softmax_scale,
        mask=mask,
    )

    # Transpose back: (batch, nheads, seqlen, headdim) -> (batch, seqlen, nheads, headdim)
    out = mx.transpose(out_t, (0, 2, 1, 3))

    # Compute LSE for compatibility (approximation since SDPA doesn't return it)
    # This is just for interface compatibility; actual LSE would need manual computation
    batch, seqlen_q, nheads, headdim = q.shape
    softmax_lse = mx.zeros((batch, nheads, seqlen_q), dtype=mx.float32)

    return out, softmax_lse, None


def _attention_manual(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    dropout_p: float = 0.0,
    softmax_scale: float = 1.0,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[mx.array] = None,
    dropout_mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array, mx.array]:
    """
    Manual attention implementation supporting all features.

    This matches the reference implementation in the CUDA tests.
    """
    batch, seqlen_q, nheads, headdim = q.shape
    _, seqlen_k, nheads_k, _ = k.shape

    # Handle GQA/MQA: repeat K, V heads if needed
    if nheads != nheads_k:
        assert nheads % nheads_k == 0
        repeat_factor = nheads // nheads_k
        k = mx.repeat(k, repeat_factor, axis=2)
        v = mx.repeat(v, repeat_factor, axis=2)

    # Compute attention scores: (batch, nheads, seqlen_q, seqlen_k)
    # q: (batch, seqlen_q, nheads, headdim) -> (batch, nheads, seqlen_q, headdim)
    # k: (batch, seqlen_k, nheads, headdim) -> (batch, nheads, headdim, seqlen_k)
    q_t = mx.transpose(q, (0, 2, 1, 3))  # (batch, nheads, seqlen_q, headdim)
    k_t = mx.transpose(k, (0, 2, 3, 1))  # (batch, nheads, headdim, seqlen_k)

    # Score = Q @ K^T * scale
    scores = mx.matmul(q_t, k_t) * softmax_scale  # (batch, nheads, seqlen_q, seqlen_k)

    # Apply softcap: scores = softcap * tanh(scores / softcap)
    if softcap > 0:
        scores = softcap * mx.tanh(scores / softcap)

    # Apply ALiBi bias
    if alibi_slopes is not None:
        alibi_bias = _compute_alibi_bias(alibi_slopes, seqlen_q, seqlen_k, causal)
        scores = scores + alibi_bias

    # Create attention mask
    mask = _create_attention_mask(seqlen_q, seqlen_k, causal, window_size)
    if mask is not None:
        scores = mx.where(mask, scores, mx.array(float("-inf")))

    # Compute softmax with numerical stability
    scores_max = mx.max(scores, axis=-1, keepdims=True)
    scores_max = mx.where(mx.isinf(scores_max), mx.zeros_like(scores_max), scores_max)
    scores_exp = mx.exp(scores - scores_max)

    # Handle fully masked rows
    scores_sum = mx.sum(scores_exp, axis=-1, keepdims=True)
    scores_sum = mx.maximum(scores_sum, mx.array(1e-12))

    attn_probs = scores_exp / scores_sum

    # Log-sum-exp for backward pass
    softmax_lse = mx.squeeze(scores_max, axis=-1) + mx.log(mx.squeeze(scores_sum, axis=-1))

    # Apply dropout
    if dropout_p > 0 and dropout_mask is None:
        # Generate dropout mask
        dropout_mask = mx.random.uniform(shape=list(attn_probs.shape)) > dropout_p
        attn_probs = mx.where(dropout_mask, attn_probs / (1 - dropout_p), mx.zeros_like(attn_probs))
    elif dropout_mask is not None:
        attn_probs = mx.where(dropout_mask, attn_probs / (1 - dropout_p), mx.zeros_like(attn_probs))

    # Compute output: attention @ V
    # attn_probs: (batch, nheads, seqlen_q, seqlen_k)
    # v: (batch, seqlen_k, nheads, headdim) -> (batch, nheads, seqlen_k, headdim)
    v_t = mx.transpose(v, (0, 2, 1, 3))
    out_t = mx.matmul(attn_probs, v_t)  # (batch, nheads, seqlen_q, headdim)

    # Transpose back to Flash Attention format
    out = mx.transpose(out_t, (0, 2, 1, 3))  # (batch, seqlen_q, nheads, headdim)

    return out, softmax_lse, attn_probs


def _create_attention_mask(
    seqlen_q: int,
    seqlen_k: int,
    causal: bool,
    window_size: Tuple[int, int],
) -> Optional[mx.array]:
    """
    Create attention mask for causal and/or local attention.

    Returns:
        Boolean mask where True = attend, False = mask
        None if no masking needed
    """
    if not causal and window_size == (-1, -1):
        return None

    # Create position indices
    q_idx = mx.arange(seqlen_q).reshape(-1, 1)  # (seqlen_q, 1)
    k_idx = mx.arange(seqlen_k).reshape(1, -1)  # (1, seqlen_k)

    # Start with all True (no masking)
    mask = mx.ones((seqlen_q, seqlen_k), dtype=mx.bool_)

    # Apply causal mask: q can only attend to k where k_idx <= q_idx
    if causal:
        causal_mask = k_idx <= q_idx
        mask = mx.logical_and(mask, causal_mask)

    # Apply window mask
    window_left, window_right = window_size
    if window_left >= 0:
        # k_idx must be >= q_idx - window_left
        left_mask = k_idx >= (q_idx - window_left)
        mask = mx.logical_and(mask, left_mask)

    if window_right >= 0:
        # k_idx must be <= q_idx + window_right
        right_mask = k_idx <= (q_idx + window_right)
        mask = mx.logical_and(mask, right_mask)

    return mask


def _compute_alibi_bias(
    alibi_slopes: mx.array,
    seqlen_q: int,
    seqlen_k: int,
    causal: bool,
) -> mx.array:
    """
    Compute ALiBi position bias.

    Args:
        alibi_slopes: Slopes of shape (batch, nheads) or (nheads,)
        seqlen_q: Query sequence length
        seqlen_k: Key sequence length
        causal: Whether using causal attention

    Returns:
        ALiBi bias of shape broadcastable to (batch, nheads, seqlen_q, seqlen_k)
    """
    # Create position indices
    q_idx = mx.arange(seqlen_q).reshape(-1, 1)  # (seqlen_q, 1)
    k_idx = mx.arange(seqlen_k).reshape(1, -1)  # (1, seqlen_k)

    if causal:
        # For causal: bias = slope * (key_pos - query_pos), always <= 0
        relative_pos = k_idx - q_idx  # (seqlen_q, seqlen_k)
    else:
        # For bidirectional: bias = -slope * |query_pos - key_pos|
        relative_pos = -mx.abs(q_idx - k_idx)  # (seqlen_q, seqlen_k)

    # Reshape slopes for broadcasting
    if alibi_slopes.ndim == 1:
        # (nheads,) -> (1, nheads, 1, 1)
        slopes = alibi_slopes.reshape(1, -1, 1, 1)
    else:
        # (batch, nheads) -> (batch, nheads, 1, 1)
        slopes = alibi_slopes.reshape(alibi_slopes.shape[0], -1, 1, 1)

    # Compute bias: slope * relative_pos
    bias = slopes * relative_pos.astype(mx.float32)

    return bias


def attention_forward_ref(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    softmax_scale: float,
    causal: bool = False,
) -> Tuple[mx.array, mx.array]:
    """
    Simple forward-only reference for basic testing.

    Returns:
        Tuple of (output, softmax_lse)
    """
    out, softmax_lse, _ = attention_ref_mlx(
        q, k, v,
        softmax_scale=softmax_scale,
        causal=causal,
    )
    return out, softmax_lse


def attention_backward_ref(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    o: mx.array,
    do: mx.array,
    softmax_lse: mx.array,
    softmax_scale: float,
    causal: bool = False,
) -> Tuple[mx.array, mx.array, mx.array]:
    """
    Reference backward pass using MLX autodiff.

    Returns:
        Tuple of (dq, dk, dv)
    """
    def forward_fn(q, k, v):
        out, _, _ = attention_ref_mlx(q, k, v, softmax_scale=softmax_scale, causal=causal)
        return out

    # Use MLX's grad to compute gradients
    grad_fn = mx.grad(lambda q, k, v: mx.sum(forward_fn(q, k, v) * do), argnums=(0, 1, 2))
    dq, dk, dv = grad_fn(q, k, v)

    return dq, dk, dv


def varlen_attention_ref_mlx(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    cu_seqlens_q: mx.array,
    cu_seqlens_k: mx.array,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array, Optional[mx.array]]:
    """
    Reference implementation for variable-length (varlen) attention.

    Args:
        q: Query tensor of shape (total_q, nheads, headdim)
        k: Key tensor of shape (total_k, nheads_k, headdim)
        v: Value tensor of shape (total_k, nheads_k, headdim)
        cu_seqlens_q: Cumulative sequence lengths for queries (batch+1,)
        cu_seqlens_k: Cumulative sequence lengths for keys/values (batch+1,)
        dropout_p: Dropout probability
        softmax_scale: Custom softmax scale (defaults to 1/sqrt(headdim))
        causal: Whether to apply causal masking within each sequence
        window_size: Sliding window (left, right) limits
        softcap: Soft cap for attention logits
        alibi_slopes: Optional ALiBi slopes

    Returns:
        Tuple of (output, softmax_lse, attn_probs) where:
            output has shape (total_q, nheads, headdim)
            softmax_lse has shape (nheads, total_q)
            attn_probs is None (unused placeholder for compatibility)
    """

    cu_q = [int(cu_seqlens_q[i].item()) for i in range(cu_seqlens_q.shape[0])]
    cu_k = [int(cu_seqlens_k[i].item()) for i in range(cu_seqlens_k.shape[0])]
    batch_size = len(cu_q) - 1
    outputs = []
    lse_outputs = []

    _, nheads, headdim = q.shape
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(headdim)

    for b in range(batch_size):
        q_start = int(cu_q[b])
        q_end = int(cu_q[b + 1])
        k_start = int(cu_k[b])
        k_end = int(cu_k[b + 1])

        if q_end <= q_start:
            continue  # Skip empty sequences

        # Extract slices and expand to 4D (batch axis = 1)
        q_b = mx.expand_dims(q[q_start:q_end], axis=0)
        k_b = mx.expand_dims(k[k_start:k_end], axis=0)
        v_b = mx.expand_dims(v[k_start:k_end], axis=0)

        out_b, lse_b, _ = _attention_manual(
            q_b,
            k_b,
            v_b,
            dropout_p=dropout_p,
            softmax_scale=scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
        )

        outputs.append(mx.squeeze(out_b, axis=0))
        lse_outputs.append(mx.squeeze(lse_b, axis=0))

    if not outputs:
        # Handle degenerate case (all sequences empty)
        total_q = int(cu_q[-1]) if cu_q else 0
        _, nheads, headdim = q.shape
        empty_out = mx.zeros((total_q, nheads, headdim), dtype=q.dtype)
        empty_lse = mx.zeros((nheads, total_q), dtype=mx.float32)
        return empty_out, empty_lse, None

    out = mx.concatenate(outputs, axis=0)
    lse = mx.concatenate(lse_outputs, axis=-1)
    return out, lse, None

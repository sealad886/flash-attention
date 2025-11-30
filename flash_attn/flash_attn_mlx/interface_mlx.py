"""
MLX Flash Attention Interface

This module provides the high-level Python interface for Flash Attention on MLX,
matching the API signatures of the CUDA backend in flash_attn_interface.py.
"""

import math
from typing import Optional, Tuple, Union

import mlx.core as mx

from flash_attn.flash_attn_mlx.utils import (
    DEBUG,
    USE_REF,
    check_args,
    maybe_contiguous,
    validate_paged_kv_params,
)
from flash_attn.flash_attn_mlx.reference import (
    attention_ref_mlx,
    varlen_attention_ref_mlx,
)
from flash_attn.flash_attn_mlx.device import is_mlx_available, get_gpu_family
from flash_attn.flash_attn_mlx.ops import flash_attention_mlx, flash_attention_with_lse, _state as _ops_state
from flash_attn.flash_attn_mlx.varlen_ops import varlen_flash_attention_mlx
from flash_attn.flash_attn_mlx.paged_cache import update_paged_kv_cache


def _resolve_philox_state(dropout_p: float) -> Tuple[int, int]:
    """Return Philox RNG seed/offset pair for dropout."""
    if dropout_p <= 0.0:
        return 0, 0
    return 0x1BF58, 0x1D4B49


def _normalize_window_size(window_size: Tuple[int, int]) -> Tuple[int, int]:
    """Validate and normalize window_size tuples to integers."""
    if len(window_size) != 2:
        raise ValueError("window_size must be a tuple of (left, right)")

    window_left, window_right = window_size
    window_left = int(window_left)
    window_right = int(window_right)

    if window_left < -1 or window_right < -1:
        raise ValueError("window_size values must be >= -1")

    return window_left, window_right


def _infer_max_seqlen(cu_seqlens: mx.array) -> int:
    """Infer the maximum segment length from a cu_seqlens array."""
    if cu_seqlens.shape[0] <= 1:
        return 0
    diffs = cu_seqlens[1:] - cu_seqlens[:-1]
    if diffs.shape[0] == 0:
        return 0
    return int(mx.max(diffs).item())


def _normalize_cache_seqlens(
    cache_seqlens: Optional[Union[int, mx.array]],
    batch: int,
) -> mx.array:
    """Return a per-batch cache length vector in int32."""

    if cache_seqlens is None:
        return mx.zeros((batch,), dtype=mx.int32)

    if isinstance(cache_seqlens, int):
        if cache_seqlens < 0:
            raise ValueError("cache_seqlens integer values must be non-negative")
        return mx.full((batch,), cache_seqlens, dtype=mx.int32)

    cache_arr = mx.array(cache_seqlens, dtype=mx.int32)
    if cache_arr.shape != (batch,):
        raise ValueError(
            f"cache_seqlens must have shape ({batch},), got {cache_arr.shape}"
        )
    return cache_arr


def flash_attn_func(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[mx.array] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> Union[mx.array, Tuple[mx.array, mx.array, mx.array]]:
    """
    Compute scaled dot-product attention using Flash Attention algorithm.

    This is the main entry point for Flash Attention on MLX backend.

    Args:
        q: Query tensor of shape (batch, seqlen_q, nheads, headdim)
        k: Key tensor of shape (batch, seqlen_k, nheads_k, headdim)
        v: Value tensor of shape (batch, seqlen_k, nheads_k, headdim)
        dropout_p: Dropout probability applied in the Metal kernels. Values > 0
            enable deterministic Philox-based dropout masks that align with the
            reference implementation.
        softmax_scale: Scaling factor for attention scores. Default: 1/sqrt(headdim)
        causal: If True, apply causal mask. Default: False
        window_size: Tuple (left, right) for sliding window attention.
            (-1, -1) means full attention. Default: (-1, -1)
        softcap: Soft cap for attention logits (0.0 means disabled). Default: 0.0
        alibi_slopes: ALiBi slopes of shape (batch, nheads) or (nheads,). The
            Metal kernels broadcast these slopes per head to add linear biases
            without falling back to the reference path.
        deterministic: If True, use deterministic backward. Default: False
        return_attn_probs: If True, return attention probabilities. Default: False

    Returns:
        output: Attention output of shape (batch, seqlen_q, nheads, headdim)

        If return_attn_probs is True:
            (output, softmax_lse, attn_probs) where:
            - softmax_lse: Log-sum-exp of attention scores (batch, nheads, seqlen_q)
            - attn_probs: Attention probabilities (only for debugging)

    Note:
        - nheads_k must divide nheads (for GQA/MQA support)
        - headdim must be one of: 32, 64, 96, 128, 160, 192, 256
    """
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])

    q, k, v = maybe_contiguous(q), maybe_contiguous(k), maybe_contiguous(v)
    check_args(q, k, v)

    if DEBUG:
        print()
        print("flash_attn_mlx.py::flash_attn_func inputs")
        print("q:", q.shape, q.dtype)
        print("k:", k.shape, k.dtype)
        print("v:", v.shape, v.dtype)
        print("dropout_p:", dropout_p)
        print("softmax_scale:", softmax_scale)
        print("causal:", causal)
        print("window_size:", window_size)
        print("softcap:", softcap)
        print("alibi_slopes:", alibi_slopes.shape if alibi_slopes is not None else None)

    if USE_REF:
        if DEBUG:
            print("Using reference implementation")
        out, softmax_lse, attn_probs = attention_ref_mlx(
            q, k, v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
        )
    else:
        if DEBUG:
            print("Using Metal kernel implementation")

        # Metal kernel now supports ALiBi and dropout - no fallback needed
        needs_fallback = False

        if needs_fallback:
            if DEBUG:
                print("Falling back to reference (feature not yet in Metal kernel)")
            out, softmax_lse, attn_probs = attention_ref_mlx(
                q, k, v,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
                alibi_slopes=alibi_slopes,
            )
        else:
            # Generate philox seed for dropout reproducibility
            philox_seed, philox_offset = _resolve_philox_state(dropout_p)

            # Use flash_attention_mlx for autograd support (VJP registered)
            out = flash_attention_mlx(
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
            # Retrieve LSE from ops state (set during forward)
            softmax_lse = _ops_state.lse
            attn_probs = None

    if DEBUG:
        print("flash_attn_mlx.py::flash_attn_func outputs")
        print("out:", out.shape, out.dtype)
        print("softmax_lse:", softmax_lse.shape if softmax_lse is not None else None)

    if return_attn_probs:
        if attn_probs is None:
            attn_probs = mx.zeros((q.shape[0], q.shape[2], q.shape[1], k.shape[1]), dtype=q.dtype)
        if softmax_lse is None:
            softmax_lse = mx.zeros((q.shape[0], q.shape[2], q.shape[1]), dtype=mx.float32)
        return out, softmax_lse, attn_probs
    return out


def flash_attn_qkvpacked_func(
    qkv: mx.array,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[mx.array] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> Union[mx.array, Tuple[mx.array, mx.array, mx.array]]:
    """
    Compute attention with packed QKV tensor.

    Args:
        qkv: Packed QKV tensor of shape (batch, seqlen, 3, nheads, headdim)
        dropout_p: Dropout probability applied via Philox-based masks that stay
            deterministic across forward/backward passes.
        softmax_scale: Scaling factor for attention scores
        causal: If True, apply causal mask
        window_size: Tuple (left, right) for sliding window attention
        softcap: Soft cap for attention logits
        alibi_slopes: ALiBi slopes of shape (batch, nheads) or (nheads,)
            broadcast per head inside the Metal kernels.
        deterministic: If True, use deterministic backward
        return_attn_probs: If True, return attention probabilities

    Returns:
        Same as flash_attn_func
    """
    batch, seqlen, _, nheads, headdim = qkv.shape
    q = qkv[:, :, 0, :, :]
    k = qkv[:, :, 1, :, :]
    v = qkv[:, :, 2, :, :]

    return flash_attn_func(
        q, k, v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
    )


def flash_attn_kvpacked_func(
    q: mx.array,
    kv: mx.array,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[mx.array] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> Union[mx.array, Tuple[mx.array, mx.array, mx.array]]:
    """
    Compute attention with separate Q and packed KV tensors.

    Args:
        q: Query tensor of shape (batch, seqlen_q, nheads, headdim)
        kv: Packed KV tensor of shape (batch, seqlen_k, 2, nheads_k, headdim)
        dropout_p: Dropout probability applied via Philox-based masks that stay
            deterministic across forward/backward passes.
        softmax_scale: Scaling factor for attention scores
        causal: If True, apply causal mask
        window_size: Tuple (left, right) for sliding window attention
        softcap: Soft cap for attention logits
        alibi_slopes: ALiBi slopes of shape (batch, nheads) or (nheads,)
            broadcast per head inside the Metal kernels.
        deterministic: If True, use deterministic backward
        return_attn_probs: If True, return attention probabilities

    Returns:
        Same as flash_attn_func
    """
    batch, seqlen_k, _, nheads_k, headdim = kv.shape
    k = kv[:, :, 0, :, :]
    v = kv[:, :, 1, :, :]

    return flash_attn_func(
        q, k, v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
    )


def flash_attn_varlen_func(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    cu_seqlens_q: mx.array,
    cu_seqlens_k: mx.array,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[mx.array] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    block_table: Optional[mx.array] = None,
) -> Union[mx.array, Tuple[mx.array, mx.array, mx.array]]:
    """Variable-length Flash Attention for packed/ragged sequences.

    This function computes attention over sequences with different lengths
    without padding overhead. Sequences are concatenated into a single tensor
    and indexed via cumulative sequence length arrays.

    Args:
        q: Query tensor of shape ``(total_q, nheads, headdim)`` where
            ``total_q = cu_seqlens_q[-1]`` is the sum of all query lengths.
        k: Key tensor of shape ``(total_k, nheads_k, headdim)``. When
            ``block_table`` is provided, pass the paged cache tensor with shape
            ``(num_pages, page_size, nheads_k, headdim)`` instead.
        v: Value tensor matching ``k`` shape.
        cu_seqlens_q: Cumulative sequence lengths for Q, shape ``(batch + 1,)``,
            dtype int32. ``cu_seqlens_q[i]`` is the start index for batch element i.
            ``cu_seqlens_q[i+1] - cu_seqlens_q[i]`` is the sequence length for batch i.
        cu_seqlens_k: Cumulative sequence lengths for K/V, same format as ``cu_seqlens_q``.
        max_seqlen_q: Maximum sequence length in Q (for kernel dispatch optimization).
        max_seqlen_k: Maximum sequence length in K/V.
        dropout_p: Dropout probability applied in Metal kernels (0 disables)
            using deterministic Philox RNG shared with the backward pass.
        softmax_scale: Scaling factor for attention scores. Default: 1/sqrt(headdim).
        causal: Apply causal masking when True. Each query at position i can only
            attend to keys at positions <= i within its own sequence.
        window_size: Tuple ``(left, right)`` for sliding window attention. Use -1
            for unbounded. E.g., ``(64, 0)`` means each query attends to the
            previous 64 keys and the current key only.
        softcap: Soft cap for logits (0 = disabled). Applies tanh-based capping:
            ``softcap * tanh(logits / softcap)``.
        alibi_slopes: ALiBi slopes of shape ``(nheads,)`` or ``(batch, nheads)``.
            Applies position-based bias ``slope * (key_pos - query_pos)`` with
            full Metal kernel support (no reference fallback).
        deterministic: Reserved for CUDA parity (MLX Metal kernels are always
            deterministic by design).
        return_attn_probs: When True, also return attention probabilities (softmax LSE).
        block_table: Optional int32 page table ``(batch, max_pages_per_seq)``
            enabling paged KV caches. Requires ``k``/``v`` to be paged tensors.

    Returns:
        Output tensor of shape ``(total_q, nheads, headdim)``.
        If ``return_attn_probs=True``, returns tuple of (output, lse, attn_probs).

    Example:
        >>> import mlx.core as mx
        >>> # Batch of 3 sequences with lengths [10, 5, 15]
        >>> seqlens = [10, 5, 15]
        >>> total_tokens = sum(seqlens)  # 30
        >>> cu_seqlens = mx.array([0, 10, 15, 30], dtype=mx.int32)
        >>>
        >>> nheads, nheads_k, headdim = 8, 8, 64
        >>> q = mx.random.normal((total_tokens, nheads, headdim))
        >>> k = mx.random.normal((total_tokens, nheads_k, headdim))
        >>> v = mx.random.normal((total_tokens, nheads_k, headdim))
        >>>
        >>> out = flash_attn_varlen_func(
        ...     q, k, v,
        ...     cu_seqlens_q=cu_seqlens,
        ...     cu_seqlens_k=cu_seqlens,
        ...     max_seqlen_q=max(seqlens),
        ...     max_seqlen_k=max(seqlens),
        ...     causal=True,
        ... )
        >>> out.shape
        (30, 8, 64)

    Note:
        For GQA/MQA, ``nheads`` must be divisible by ``nheads_k``. Each group
        of ``nheads // nheads_k`` query heads shares one key-value head.
    """
    window_left, window_right = _normalize_window_size(window_size)
    window_size = (window_left, window_right)

    paged_kv_enabled = block_table is not None

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])

    q = maybe_contiguous(q)
    k = maybe_contiguous(k)
    v = maybe_contiguous(v)

    k_tensor = k
    v_tensor = v
    page_table = None

    cu_seqlens_q = mx.array(cu_seqlens_q, dtype=mx.int32)
    cu_seqlens_k = mx.array(cu_seqlens_k, dtype=mx.int32)

    if cu_seqlens_q.shape[0] < 2 or cu_seqlens_k.shape[0] < 2:
        raise ValueError("cu_seqlens arrays must have at least two elements")

    if cu_seqlens_q.shape[0] != cu_seqlens_k.shape[0]:
        raise ValueError("cu_seqlens_q and cu_seqlens_k must have the same length")

    total_q = int(cu_seqlens_q[-1].item())
    total_k = int(cu_seqlens_k[-1].item())

    if q.shape[0] != total_q:
        raise ValueError(
            f"Mismatch between q tokens ({q.shape[0]}) and cu_seqlens_q total ({total_q})"
        )
    if q.ndim != 3:
        raise ValueError("Varlen query tensor must have shape (total_tokens, nheads, headdim)")

    if paged_kv_enabled:
        page_table = mx.array(block_table, dtype=mx.int32)
        if page_table.ndim != 2:
            raise ValueError(f"block_table must be 2D, got {page_table.ndim}D")
        batch = int(cu_seqlens_q.shape[0]) - 1
        if page_table.shape[0] != batch:
            raise ValueError(
                f"block_table batch dimension ({page_table.shape[0]}) must match sequences ({batch})",
            )
        if k_tensor.ndim != 4 or v_tensor.ndim != 4:
            raise ValueError(
                "Paged KV tensors must be 4D (num_pages, page_size, nheads_k, headdim)",
            )
        if k_tensor.shape != v_tensor.shape:
            raise ValueError("Paged K/V caches must have matching shapes")

        page_size, max_pages_per_seq = validate_paged_kv_params(
            k_cache=k_tensor,
            v_cache=v_tensor,
            page_table=page_table,
            batch_size=batch,
        )
        num_pages = int(k_tensor.shape[0])
        total_physical_capacity = num_pages * page_size
        if total_k > total_physical_capacity:
            raise ValueError(
                f"cu_seqlens_k total ({total_k}) exceeds paged cache physical capacity "
                f"({total_physical_capacity} = {num_pages} pages × {page_size} tokens/page)",
            )

    else:
        if k_tensor.shape[0] != total_k or v_tensor.shape[0] != total_k:
            raise ValueError("Mismatch between K/V tokens and cu_seqlens_k total")
        if k_tensor.shape != v_tensor.shape:
            raise ValueError("Packed K and V must have matching shapes in varlen mode")
        if k_tensor.ndim != 3 or v_tensor.ndim != 3:
            raise ValueError(
                "Varlen K/V tensors must have shape (total_tokens, nheads, headdim)",
            )

    nheads = q.shape[1]
    nheads_k = k_tensor.shape[-2]
    if nheads_k == 0 or nheads % nheads_k != 0:
        raise ValueError("nheads must be divisible by nheads_k for varlen attention")

    if max_seqlen_q is None:
        max_seqlen_q = _infer_max_seqlen(cu_seqlens_q)
    if max_seqlen_k is None:
        max_seqlen_k = _infer_max_seqlen(cu_seqlens_k)

    if DEBUG:
        print()
        print("flash_attn_mlx.py::flash_attn_varlen_func inputs")
        print("q:", q.shape, q.dtype)
        print("k:", k_tensor.shape, k_tensor.dtype)
        print("v:", v_tensor.shape, v_tensor.dtype)
        print("cu_seqlens_q:", cu_seqlens_q.shape)
        print("cu_seqlens_k:", cu_seqlens_k.shape)
        print("max_seqlen_q:", max_seqlen_q)
        print("max_seqlen_k:", max_seqlen_k)
        print("softmax_scale:", softmax_scale)
        print("causal:", causal)
        print("window_size:", window_size)
        print("softcap:", softcap)

    use_reference = USE_REF

    if paged_kv_enabled and use_reference:
        raise RuntimeError(
            "Paged KV cache is only available via the Metal kernel; disable USE_REF",
        )

    if use_reference:
        out, softmax_lse, attn_probs = varlen_attention_ref_mlx(
            q,
            k_tensor,
            v_tensor,
            cu_seqlens_q,
            cu_seqlens_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
        )
    else:
        philox_seed, philox_offset = _resolve_philox_state(dropout_p)
        out, softmax_lse = varlen_flash_attention_mlx(
            q,
            k_tensor,
            v_tensor,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            use_metal_kernel=True,
            alibi_slopes=alibi_slopes,
            page_table=page_table,
            dropout_p=dropout_p,
            philox_seed=philox_seed,
            philox_offset=philox_offset,
        )
        attn_probs = None

    if return_attn_probs:
        if attn_probs is None:
            attn_probs = mx.zeros((0,), dtype=out.dtype)
        if softmax_lse is None:
            softmax_lse = mx.zeros((out.shape[1], out.shape[0]), dtype=mx.float32)
        return out, softmax_lse, attn_probs
    return out


def flash_attn_varlen_qkvpacked_func(
    qkv: mx.array,
    cu_seqlens: mx.array,
    max_seqlen: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[mx.array] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> Union[mx.array, Tuple[mx.array, mx.array, mx.array]]:
    """Compute variable-length attention when Q/K/V are packed together.

    This convenience wrapper accepts a QKV tensor where the second dimension
    enumerates Q, K, and V in order and forwards the unpacked tensors to
    :func:`flash_attn_varlen_func`. The packed layout matches the format used
    by CUDA tests as well as the helper provided in
    ``flash_attn.flash_attn_mlx.tests.varlen_test_utils.make_varlen_qkv``.

    Args:
        qkv: Packed tensor of shape ``(total_tokens, 3, nheads, headdim)`` where
            ``qkv[:, 0]`` stores queries, ``qkv[:, 1]`` stores keys, and
            ``qkv[:, 2]`` stores values in the varlen layout.
        cu_seqlens: Int32 cumulative sequence lengths with shape
            ``(batch + 1,)`` describing how tokens are partitioned per batch.
        max_seqlen: Maximum sequence length across the batch. This dispatch
            hint is forwarded to the Metal kernel for grid sizing.
        dropout_p: Dropout probability applied in the Metal kernel (0 disables dropout).
        softmax_scale: Scaling factor (defaults to ``1/sqrt(headdim)`` when
            ``None``).
        causal: Apply causal masking when ``True``.
        window_size: Sliding-window tuple ``(left, right)``; use ``(-1, -1)``
            for full attention.
        softcap: Optional logit capping factor.
        alibi_slopes: Optional ALiBi slopes broadcast to each head.
        deterministic: Reserved for CUDA parity; MLX kernels are deterministic.
        return_attn_probs: When ``True``, also return the log-sum-exp tensor and
            attention probabilities for debugging.

    Returns:
        Same as :func:`flash_attn_varlen_func`.

    Example:
        >>> seqlens = [12, 4]
        >>> q, k, v, cu_q, _, max_q, _ = make_varlen_qkv(seqlens, 8, 8, 64)
        >>> qkv = mx.stack([q, k, v], axis=1)
        >>> out = flash_attn_varlen_qkvpacked_func(qkv, cu_q, max_q)
    """
    total, _, nheads, headdim = qkv.shape
    q = qkv[:, 0, :, :]
    k = qkv[:, 1, :, :]
    v = qkv[:, 2, :, :]

    return flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
    )


def flash_attn_varlen_kvpacked_func(
    q: mx.array,
    kv: mx.array,
    cu_seqlens_q: mx.array,
    cu_seqlens_k: mx.array,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[mx.array] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> Union[mx.array, Tuple[mx.array, mx.array, mx.array]]:
    """Variant of :func:`flash_attn_varlen_func` for packed KV inputs.

    Args mirror :func:`flash_attn_varlen_func` except that keys and values are
    provided as a single tensor. The standard packed layout is
    ``(total_k, 2, nheads_k, headdim)`` where ``kv[:, 0]`` stores keys and
    ``kv[:, 1]`` stores values. When benchmarking paged caches the packed
    tensor can include an explicit page dimension, e.g.
    ``(num_pages, page_size, 2, nheads_k, headdim)``, and the helper forwards
    each slice with the supplied ``block_table``.

    Args:
        q: Packed query tensor shaped ``(total_q, nheads, headdim)``.
        kv: Packed or paged KV tensor as described above.
        cu_seqlens_q / cu_seqlens_k: Int32 cumulative sequence lengths of shape
            ``(batch + 1,)`` describing how tokens are concatenated.
        max_seqlen_q / max_seqlen_k: Maximum lengths used to configure the
            Metal launch grid.
        dropout_p, softmax_scale, causal, window_size, softcap, alibi_slopes,
            deterministic, return_attn_probs: Same semantics as
            :func:`flash_attn_varlen_func`.

    Returns:
        Same as :func:`flash_attn_varlen_func`.

    Example:
        >>> q, k, v, cu_q, cu_k, max_q, max_k = make_varlen_qkv([16, 7], 8, 2, 64)
        >>> kv = mx.stack([k, v], axis=1)
        >>> out = flash_attn_varlen_kvpacked_func(q, kv, cu_q, cu_k, max_q, max_k)
    """
    total_k, _, nheads_k, headdim = kv.shape
    k = kv[:, 0, :, :]
    v = kv[:, 1, :, :]

    return flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
    )


def flash_attn_with_kvcache(
    q: mx.array,
    k_cache: mx.array,
    v_cache: mx.array,
    k: Optional[mx.array] = None,
    v: Optional[mx.array] = None,
    rotary_cos: Optional[mx.array] = None,
    rotary_sin: Optional[mx.array] = None,
    cache_seqlens: Optional[Union[int, mx.array]] = None,
    cache_batch_idx: Optional[mx.array] = None,
    cache_leftpad: Optional[mx.array] = None,
    block_table: Optional[mx.array] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    rotary_interleaved: bool = True,
    alibi_slopes: Optional[mx.array] = None,
    num_splits: int = 0,
    return_softmax_lse: bool = False,
    return_cache_seqlens: bool = False,
) -> Union[mx.array, Tuple[mx.array, ...]]:
    """Compute attention using a (potentially paged) KV cache for decoding.

    This API mirrors ``flash_attn_interface.flash_attn_with_kvcache`` on CUDA.
    When ``k``/``v`` are provided, the cache is updated before issuing the
    attention query. In paged mode the cache is addressed indirectly via a
    block/page table, matching the contract from ``hopper/paged_kv.h``.

    Args:
        q: Query tensor of shape ``(batch, seqlen_q, nheads, headdim)``.
        k_cache: Key cache. Contiguous caches use
            ``(batch, max_seqlen_k, nheads_k, headdim)`` while paged caches use
            ``(num_pages, page_size, nheads_k, headdim)``.
        v_cache: Value cache with the same layout as ``k_cache``.
        k: Optional new keys of shape ``(batch, seqlen_new, nheads_k, headdim)``
            to append at ``cache_seqlens``. Only supported when
            ``cache_seqlens`` is a scalar or when paged mode is active.
        v: Optional new values that mirror ``k``.
        rotary_cos: Rotary embedding cosines (placeholder, not applied by the
            MLX backend yet).
        rotary_sin: Rotary embedding sines (placeholder, not applied yet).
        cache_seqlens: Either a scalar int (uniform cache length) or an
            ``(batch,)`` int32 vector describing the logical number of cached
            tokens per batch element.
        cache_batch_idx: Optional integer indices enabling sliced cache access
            (feature reserved for parity with CUDA, currently unused).
        cache_leftpad: Optional per-batch left padding lengths (reserved).
        block_table: Optional int32 page table of shape
            ``(batch, max_pages_per_seq)``. Supplying this enables paged KV
            access and requires ``k_cache``/``v_cache`` to be 4-D paged tensors.
            Entries containing ``-1`` are skipped, allowing sparse page usage.
        softmax_scale: Optional scaling factor (defaults to ``1/sqrt(headdim)``).
        causal: Apply causal masking when True.
        window_size: Two-element tuple ``(left, right)`` for sliding windows
            (``(-1, -1)`` disables the window).
        softcap: Optional soft cap for attention logits.
        rotary_interleaved: Included for signature parity; rotary embeddings
            must still be applied by the caller.
        alibi_slopes: Optional ALiBi slopes shaped ``(nheads,)`` or
            ``(batch, nheads)``.
        num_splits: Split-KV hint (kept for compatibility, unused on MLX).
        return_softmax_lse: When True, include the log-sum-exp tensor in the
            output tuple.
        return_cache_seqlens: When True, append the updated cache length vector
            (after the LSE tensor if requested).

    Returns:
        Either the attention output or a tuple ``(out, lse, cache_lens)``
        depending on the ``return_*`` flags. ``out`` always has shape
        ``(batch, seqlen_q, nheads, headdim)``; ``cache_lens`` matches the
        format emitted by :func:`_normalize_cache_seqlens`.

    Raises:
        ValueError: If tensor shapes or block/table dimensions do not match.
        NotImplementedError: When attempting to write a contiguous cache with
            per-batch ``cache_seqlens`` (feature parity gap with CUDA).
    """
    batch, seqlen_q, nheads, headdim = q.shape
    _, max_seqlen_k, nheads_k, _ = k_cache.shape

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(headdim)

    if DEBUG:
        print()
        print("flash_attn_mlx.py::flash_attn_with_kvcache inputs")
        print("q:", q.shape, q.dtype)
        print("k_cache:", k_cache.shape, k_cache.dtype)
        print("v_cache:", v_cache.shape, v_cache.dtype)
        print("k:", k.shape if k is not None else None)
        print("v:", v.shape if v is not None else None)
        print("cache_seqlens:", cache_seqlens)

    cache_seqlens_arr = _normalize_cache_seqlens(cache_seqlens, batch)
    cache_seqlens_result = cache_seqlens_arr
    cache_seqlens_is_int = isinstance(cache_seqlens, int) or cache_seqlens is None

    paged_kv_enabled = block_table is not None

    if paged_kv_enabled:
        page_table = mx.array(block_table, dtype=mx.int32)
        page_size, max_pages_per_seq = validate_paged_kv_params(
            k_cache,
            v_cache,
            page_table,
            batch,
        )

        if DEBUG:
            print("Paged KV cache enabled")
            print("page_size:", page_size)
            print("max_pages_per_seq:", max_pages_per_seq)

        if k is not None and v is not None:
            cache_seqlens_result = update_paged_kv_cache(
                k_cache,
                v_cache,
                k,
                v,
                page_table,
                cache_seqlens_arr,
            )
        else:
            cache_seqlens_result = cache_seqlens_arr

        out, lse = flash_attention_with_lse(
            q, k_cache, v_cache,
            softmax_scale=softmax_scale,
            causal=causal,
            softcap=softcap,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            block_table=page_table,
        )
    else:
        cache_seqlens_scalar = int(cache_seqlens_arr[0].item()) if batch > 0 else 0

        if k is not None and v is not None:
            seqlen_new = k.shape[1]

            if cache_seqlens_is_int:
                start_pos = cache_seqlens_scalar
                end_pos = start_pos + seqlen_new

                if start_pos > 0:
                    k_cache = mx.concatenate([
                        k_cache[:, :start_pos, :, :],
                        k,
                        k_cache[:, end_pos:, :, :],
                    ], axis=1)
                    v_cache = mx.concatenate([
                        v_cache[:, :start_pos, :, :],
                        v,
                        v_cache[:, end_pos:, :, :],
                    ], axis=1)
                else:
                    k_cache = mx.concatenate([k, k_cache[:, seqlen_new:, :, :]], axis=1)
                    v_cache = mx.concatenate([v, v_cache[:, seqlen_new:, :, :]], axis=1)

                new_seqlen = cache_seqlens_scalar + seqlen_new
                cache_seqlens_result = cache_seqlens_arr + seqlen_new
            else:
                raise NotImplementedError(
                    "Variable cache_seqlens per batch not yet supported for writes. "
                    "Use a uniform integer cache length when updating the contiguous cache.",
                )
        else:
            seqlen_new = 0
            cache_seqlens_result = cache_seqlens_arr
            if cache_seqlens_is_int:
                new_seqlen = cache_seqlens_scalar
            else:
                new_seqlen = int(mx.max(cache_seqlens_arr).item())

        if cache_seqlens_is_int:
            k_active = k_cache[:, :new_seqlen, :, :]
            v_active = v_cache[:, :new_seqlen, :, :]
        else:
            max_len = int(mx.max(cache_seqlens_result).item())
            k_active = k_cache[:, :max_len, :, :]
            v_active = v_cache[:, :max_len, :, :]

        out, lse = flash_attention_with_lse(
            q, k_active, v_active,
            softmax_scale=softmax_scale,
            causal=causal,
            softcap=softcap,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
        )

    if DEBUG:
        print("flash_attn_mlx.py::flash_attn_with_kvcache outputs")
        print("out:", out.shape, out.dtype)

    if return_softmax_lse and return_cache_seqlens:
        return out, lse, cache_seqlens_result
    if return_softmax_lse:
        return out, lse
    if return_cache_seqlens:
        return out, cache_seqlens_result
    return out

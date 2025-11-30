"""Varlen Flash Attention operations with MLX autograd support."""

from __future__ import annotations

from typing import Optional, Tuple, cast

import mlx.core as mx

from .varlen_fwd_kernel import varlen_flash_attention_forward
from .varlen_bwd_kernel import varlen_flash_attention_backward


class _VarlenFlashAttentionState:
    """State container shared between forward and backward passes."""

    def __init__(self) -> None:
        self.lse: Optional[mx.array] = None
        self.cu_seqlens_q: Optional[mx.array] = None
        self.cu_seqlens_k: Optional[mx.array] = None
        self.max_seqlen_q: Optional[int] = None
        self.max_seqlen_k: Optional[int] = None
        self.softmax_scale: float = 1.0
        self.causal: bool = False
        self.window_size: Tuple[int, int] = (-1, -1)
        self.softcap: float = 0.0
        self.use_metal_kernel: bool = True
        self.alibi_slopes: Optional[mx.array] = None
        self.page_table: Optional[mx.array] = None
        self.dropout_p: float = 0.0
        self.philox_seed: int = 0
        self.philox_offset: int = 0

    def reset_runtime_state(self) -> None:
        """Reset values populated during the forward call."""
        self.lse = None
        self.cu_seqlens_q = None
        self.cu_seqlens_k = None


_STATE = _VarlenFlashAttentionState()


@mx.custom_function
def _varlen_flash_attention_core(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    cu_seqlens_q: mx.array,
    cu_seqlens_k: mx.array,
) -> mx.array:
    """Core varlen Flash Attention custom function."""
    global _STATE

    out, lse = varlen_flash_attention_forward(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q=_STATE.max_seqlen_q,
        max_seqlen_k=_STATE.max_seqlen_k,
        softmax_scale=_STATE.softmax_scale,
        causal=_STATE.causal,
        window_size=_STATE.window_size,
        softcap=_STATE.softcap,
        alibi_slopes=_STATE.alibi_slopes,
        dropout_p=_STATE.dropout_p,
        philox_seed=_STATE.philox_seed,
        philox_offset=_STATE.philox_offset,
        page_table=_STATE.page_table,
        use_metal_kernel=_STATE.use_metal_kernel,
    )
    _STATE.lse = lse
    _STATE.cu_seqlens_q = cu_seqlens_q
    _STATE.cu_seqlens_k = cu_seqlens_k
    return out


@_varlen_flash_attention_core.vjp
def _varlen_flash_attention_core_vjp(primals, cotangent, output):
    """Vector-Jacobian product for the varlen Flash Attention custom op."""
    global _STATE

    q, k, v, cu_seqlens_q, cu_seqlens_k = primals
    dout = cotangent

    if _STATE.page_table is not None:
        raise NotImplementedError(
            "Paged KV varlen attention does not currently support backward differentiation",
        )

    lse = _STATE.lse
    if lse is None:
        _, lse = varlen_flash_attention_forward(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=_STATE.max_seqlen_q,
            max_seqlen_k=_STATE.max_seqlen_k,
            softmax_scale=_STATE.softmax_scale,
            causal=_STATE.causal,
            window_size=_STATE.window_size,
            softcap=_STATE.softcap,
            alibi_slopes=_STATE.alibi_slopes,
            dropout_p=_STATE.dropout_p,
            philox_seed=_STATE.philox_seed,
            philox_offset=_STATE.philox_offset,
            page_table=_STATE.page_table,
            use_metal_kernel=False,
        )

    dq, dk, dv = varlen_flash_attention_backward(
        q,
        k,
        v,
        output,
        dout,
        lse,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q=_STATE.max_seqlen_q,
        max_seqlen_k=_STATE.max_seqlen_k,
        softmax_scale=_STATE.softmax_scale,
        causal=_STATE.causal,
        window_size=_STATE.window_size,
        softcap=_STATE.softcap,
        alibi_slopes=_STATE.alibi_slopes,
        dropout_p=_STATE.dropout_p,
        philox_seed=_STATE.philox_seed,
        philox_offset=_STATE.philox_offset,
        use_metal_kernel=_STATE.use_metal_kernel,
    )
    return dq, dk, dv, None, None


def varlen_flash_attention_mlx(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    cu_seqlens_q: mx.array,
    cu_seqlens_k: mx.array,
    max_seqlen_q: Optional[int],
    max_seqlen_k: Optional[int],
    softmax_scale: float,
    causal: bool,
    window_size: Tuple[int, int],
    softcap: float,
    use_metal_kernel: bool,
    alibi_slopes: Optional[mx.array] = None,
    page_table: Optional[mx.array] = None,
    dropout_p: float = 0.0,
    philox_seed: int = 0,
    philox_offset: int = 0,
) -> Tuple[mx.array, mx.array]:
    """Run varlen Flash Attention with Metal acceleration and autograd.

    This helper wraps the custom MLX function defined in this module, wiring up
    the runtime state required for the backward pass. It mirrors the CUDA
    :func:`flash_attn_varlen_func` entry point and is invoked internally by the
    public interface in ``interface_mlx.py``.

    Args:
        q: Packed query tensor of shape ``(total_q, nheads, headdim)``.
        k: Packed or paged key tensor. Packed layouts use
            ``(total_k, nheads_k, headdim)`` while paged caches use
            ``(num_pages, page_size, nheads_k, headdim)``.
        v: Value tensor matching ``k``'s layout.
        cu_seqlens_q: Int32 cumulative sequence lengths for queries. Must have
            ``batch + 1`` entries with ``cu_seqlens_q[0] = 0`` and
            ``cu_seqlens_q[-1] = total_q``.
        cu_seqlens_k: Same as ``cu_seqlens_q`` but for keys/values.
        max_seqlen_q / max_seqlen_k: Maximum lengths for each set of sequences
            (used to size the Metal launch grid). When ``None`` they can be
            inferred from the cu_seqlens arrays.
        softmax_scale: Scaling factor applied to attention logits. Typically
            ``1/sqrt(headdim)``.
        causal: Apply causal masking within each sequence when ``True``.
        window_size: Sliding-window tuple ``(left, right)`` measured in tokens
            relative to each query position (``-1`` disables the bound).
        softcap: Optional tanh-based logit capping factor.
        use_metal_kernel: Controls whether to call the Metal kernels or the
            reference fallback (used for debugging).
        alibi_slopes: Optional ALiBi slopes broadcast per head.
        page_table: Optional int32 block table enabling paged KV caches. Must
            be supplied when ``k``/``v`` are paged tensors.
        dropout_p: Dropout probability applied after softmax. Values greater
            than zero trigger deterministic Philox-based mask generation that
            matches the reference implementation, ensuring identical backward
            gradients.
        philox_seed / philox_offset: Explicit Philox RNG state forwarded from
            the high-level interface. These values allow the backward kernels
            to regenerate the exact same dropout mask without storing it.

    Returns:
        Tuple ``(output, softmax_lse)`` where ``output`` has shape
        ``(total_q, nheads, headdim)`` and ``softmax_lse`` stores the
        log-sum-exp tensor consumed by the backward kernels.
    """
    global _STATE

    cu_seqlens_q = mx.array(cu_seqlens_q, dtype=mx.int32)
    cu_seqlens_k = mx.array(cu_seqlens_k, dtype=mx.int32)

    _STATE.max_seqlen_q = max_seqlen_q
    _STATE.max_seqlen_k = max_seqlen_k
    _STATE.softmax_scale = softmax_scale
    _STATE.causal = causal
    _STATE.window_size = window_size
    _STATE.softcap = softcap
    _STATE.use_metal_kernel = use_metal_kernel
    _STATE.alibi_slopes = alibi_slopes
    _STATE.page_table = page_table
    _STATE.dropout_p = dropout_p
    _STATE.philox_seed = philox_seed
    _STATE.philox_offset = philox_offset
    _STATE.reset_runtime_state()

    out = cast(
        mx.array,
        _varlen_flash_attention_core(q, k, v, cu_seqlens_q, cu_seqlens_k),
    )
    lse = _STATE.lse
    if lse is None:
        # Should not happen, but fall back to reference computation.
        _, lse = varlen_flash_attention_forward(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=_STATE.max_seqlen_q,
            max_seqlen_k=_STATE.max_seqlen_k,
            softmax_scale=_STATE.softmax_scale,
            causal=_STATE.causal,
            window_size=_STATE.window_size,
            softcap=_STATE.softcap,
            alibi_slopes=_STATE.alibi_slopes,
            dropout_p=_STATE.dropout_p,
            philox_seed=_STATE.philox_seed,
            philox_offset=_STATE.philox_offset,
            page_table=_STATE.page_table,
            use_metal_kernel=False,
        )
    return out, lse

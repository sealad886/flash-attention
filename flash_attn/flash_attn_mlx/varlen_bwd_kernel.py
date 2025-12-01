"""Varlen Flash Attention backward kernel wrapper."""

from __future__ import annotations

from typing import Optional, Tuple

import mlx.core as mx

from .kernels import (
    MetalKernelSpec,
    get_attention_varlen_backward_dq_kernel,
    get_attention_varlen_backward_dkv_kernel,
    get_compiled_kernel,
    get_utils_header,
    register_metal_kernel,
)
from .params import VarlenAttentionParams, create_varlen_params
from .reference import varlen_attention_ref_mlx


_VARLEN_DQ_KERNEL_NAME = "flash_attention_varlen_bwd_dq"
_VARLEN_DKV_KERNEL_NAME = "flash_attention_varlen_bwd_dkv"

register_metal_kernel(
    MetalKernelSpec(
        name=_VARLEN_DQ_KERNEL_NAME,
        input_names=[
            "Q",
            "K",
            "V",
            "O",
            "dO",
            "L",
            "cu_seqlens_q",
            "cu_seqlens_k",
            "batch_size",
            "nheads",
            "nheads_k",
            "headdim",
            "total_q",
            "softmax_scale",
            "causal_flag",
            "softcap",
            "window_left",
            "window_right",
            "alibi_slopes",
            "use_alibi",
            "alibi_batch_stride",
            "dropout_p",
            "philox_seed",
            "philox_offset",
        ],
        output_names=["dQ"],
        source_fn=get_attention_varlen_backward_dq_kernel,
        header=get_utils_header(),
        ensure_row_contiguous=True,
    )
)

register_metal_kernel(
    MetalKernelSpec(
        name=_VARLEN_DKV_KERNEL_NAME,
        input_names=[
            "Q",
            "K",
            "V",
            "O",
            "dO",
            "L",
            "cu_seqlens_q",
            "cu_seqlens_k",
            "batch_size",
            "nheads",
            "nheads_k",
            "headdim",
            "total_q",
            "total_k",
            "softmax_scale",
            "causal_flag",
            "softcap",
            "window_left",
            "window_right",
            "alibi_slopes",
            "use_alibi",
            "alibi_batch_stride",
            "dropout_p",
            "philox_seed",
            "philox_offset",
        ],
        output_names=["dK", "dV"],
        source_fn=get_attention_varlen_backward_dkv_kernel,
        header=get_utils_header(),
        ensure_row_contiguous=True,
    )
)


def _get_varlen_dq_kernel():
    """Compile or retrieve the cached varlen dQ Metal kernel."""
    return get_compiled_kernel(_VARLEN_DQ_KERNEL_NAME)


def _get_varlen_dkv_kernel():
    """Compile or retrieve the cached varlen dK/dV Metal kernel."""
    return get_compiled_kernel(_VARLEN_DKV_KERNEL_NAME)


def _validate_cu_seqlens(array: mx.array, name: str) -> None:
    if array.dtype != mx.int32:
        raise ValueError(f"{name} must be int32 (got {array.dtype})")
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-D (got shape {array.shape})")
    if array.shape[0] < 2:
        raise ValueError(f"{name} must have at least two elements (got {array.shape[0]})")


def _varlen_backward_reference(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    dout: mx.array,
    cu_seqlens_q: mx.array,
    cu_seqlens_k: mx.array,
    softmax_scale: float,
    causal: bool,
    window_size: Tuple[int, int],
    softcap: float,
    dropout_p: float,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Reference varlen backward computation using MX gradients."""

    def _forward_fn(q_in, k_in, v_in):
        out, _, _ = varlen_attention_ref_mlx(
            q_in,
            k_in,
            v_in,
            cu_seqlens_q,
            cu_seqlens_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
        )
        return out

    grad_fn = mx.grad(
        lambda q_in, k_in, v_in: mx.sum(_forward_fn(q_in, k_in, v_in) * dout),
        argnums=(0, 1, 2),
    )
    return grad_fn(q, k, v)


def varlen_flash_attention_backward(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    o: mx.array,
    dout: mx.array,
    lse: mx.array,
    cu_seqlens_q: mx.array,
    cu_seqlens_k: mx.array,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[mx.array] = None,
    dropout_p: float = 0.0,
    philox_seed: int = 0,
    philox_offset: int = 0,
    use_metal_kernel: bool = True,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Backward pass for variable-length Flash Attention."""

    cu_seqlens_q = mx.array(cu_seqlens_q, dtype=mx.int32)
    cu_seqlens_k = mx.array(cu_seqlens_k, dtype=mx.int32)
    lse = mx.array(lse, dtype=mx.float32)
    _validate_cu_seqlens(cu_seqlens_q, "cu_seqlens_q")
    _validate_cu_seqlens(cu_seqlens_k, "cu_seqlens_k")

    params: VarlenAttentionParams = create_varlen_params(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
    )

    if params.nheads_k == 0 or params.nheads % params.nheads_k != 0:
        raise ValueError("nheads must be divisible by nheads_k for varlen attention")

    if params.total_q == 0 or params.total_k == 0 or params.batch_size == 0:
        empty_dq = mx.zeros_like(q)
        empty_dk = mx.zeros_like(k)
        empty_dv = mx.zeros_like(v)
        return empty_dq, empty_dk, empty_dv

    if not use_metal_kernel:
        return _varlen_backward_reference(
            q,
            k,
            v,
            dout,
            cu_seqlens_q,
            cu_seqlens_k,
            params.softmax_scale,
            causal,
            window_size,
            softcap,
            dropout_p,
        )

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

    dq_kernel = _get_varlen_dq_kernel()
    dkv_kernel = _get_varlen_dkv_kernel()

    dq_grid = (params.total_q * params.nheads, 1, 1)
    dq_threadgroup = (128, 1, 1)
    if dq_grid[0] < 128:
        dq_threadgroup = (64 if dq_grid[0] > 64 else 32, 1, 1)

    dkv_grid = (params.total_k * params.nheads_k, 1, 1)
    dkv_threadgroup = (128, 1, 1)
    if dkv_grid[0] < 128:
        dkv_threadgroup = (64 if dkv_grid[0] > 64 else 32, 1, 1)

    try:
        dq_outputs = dq_kernel(
            inputs=[
                q,
                k,
                v,
                o,
                dout,
                lse,
                cu_seqlens_q,
                cu_seqlens_k,
                params.batch_size,
                params.nheads,
                params.nheads_k,
                params.headdim,
                params.total_q,
                params.softmax_scale,
                causal_int,
                softcap,
                window_left,
                window_right,
                alibi_slopes_arr,
                use_alibi,
                alibi_batch_stride,
                dropout_p,
                philox_seed,
                philox_offset,
            ],
            template=[("T", q.dtype)],
            grid=dq_grid,
            threadgroup=dq_threadgroup,
            output_shapes=[q.shape],
            output_dtypes=[q.dtype],
            init_value=0.0,
        )
        dq = dq_outputs[0]

        dkv_outputs = dkv_kernel(
            inputs=[
                q,
                k,
                v,
                o,
                dout,
                lse,
                cu_seqlens_q,
                cu_seqlens_k,
                params.batch_size,
                params.nheads,
                params.nheads_k,
                params.headdim,
                params.total_q,
                params.total_k,
                params.softmax_scale,
                causal_int,
                softcap,
                window_left,
                window_right,
                alibi_slopes_arr,
                use_alibi,
                alibi_batch_stride,
                dropout_p,
                philox_seed,
                philox_offset,
            ],
            template=[("T", q.dtype)],
            grid=dkv_grid,
            threadgroup=dkv_threadgroup,
            output_shapes=[k.shape, v.shape],
            output_dtypes=[k.dtype, v.dtype],
            init_value=0.0,
        )
        dk, dv = dkv_outputs
        return dq, dk, dv

    except Exception:
        return _varlen_backward_reference(
            q,
            k,
            v,
            dout,
            cu_seqlens_q,
            cu_seqlens_k,
            params.softmax_scale,
            causal,
            window_size,
            softcap,
            dropout_p,
        )

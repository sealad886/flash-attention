"""Varlen Flash Attention forward kernel wrapper."""

from __future__ import annotations
from typing import Optional, Tuple

import mlx.core as mx

from .kernels import (
    MetalKernelSpec,
    get_attention_varlen_forward_kernel,
    get_compiled_kernel,
    get_utils_header,
    register_metal_kernel,
)
from .params import VarlenAttentionParams, create_varlen_params
from .reference import varlen_attention_ref_mlx
from .utils import log2_if_power_of_two, validate_paged_kv_params


_VARLEN_FWD_KERNEL_NAME = "flash_attention_varlen_fwd"
_VARLEN_DUMMY_PAGE_TABLE = mx.zeros((1,), dtype=mx.int32)

register_metal_kernel(
    MetalKernelSpec(
        name=_VARLEN_FWD_KERNEL_NAME,
        input_names=[
            "Q",
            "K",
            "V",
            "cu_seqlens_q",
            "cu_seqlens_k",
            "batch_size",
            "nheads",
            "nheads_k",
            "headdim",
            "total_q",
            "softmax_scale",
            "causal",
            "softcap",
            "window_left",
            "window_right",
            "alibi_slopes",
            "use_alibi",
            "alibi_batch_stride",
            "page_table",
            "page_size",
            "max_pages_per_seq",
            "use_paged_kv",
            "log2_page_size",
            "dropout_p",
            "philox_seed",
            "philox_offset",
        ],
        output_names=["O", "L"],
        source_fn=get_attention_varlen_forward_kernel,
        header=get_utils_header(),
        ensure_row_contiguous=True,
    )
)


def _get_varlen_forward_kernel():
    """Compile or retrieve the cached varlen forward Metal kernel."""
    return get_compiled_kernel(_VARLEN_FWD_KERNEL_NAME)


def _validate_cu_seqlens(array: mx.array, name: str) -> None:
    if array.dtype != mx.int32:
        raise ValueError(f"{name} must be int32 (got {array.dtype})")
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-D (got shape {array.shape})")
    if array.shape[0] < 2:
        raise ValueError(f"{name} must have at least two elements (got {array.shape[0]})")


def varlen_flash_attention_forward(
    q: mx.array,
    k: mx.array,
    v: mx.array,
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
    page_table: Optional[mx.array] = None,
    use_metal_kernel: bool = True,
) -> Tuple[mx.array, mx.array]:
    """Forward pass for variable-length Flash Attention.

    When ``page_table`` is provided, K/V caches are interpreted as paged buffers
    using the supplied metadata. Reference fallback is disabled in that mode.
    Otherwise, falls back to the reference implementation if the Metal kernel
    fails or if ``use_metal_kernel`` is False.
    """

    cu_seqlens_q = mx.array(cu_seqlens_q, dtype=mx.int32)
    cu_seqlens_k = mx.array(cu_seqlens_k, dtype=mx.int32)
    _validate_cu_seqlens(cu_seqlens_q, "cu_seqlens_q")
    _validate_cu_seqlens(cu_seqlens_k, "cu_seqlens_k")

    batch_size = int(cu_seqlens_q.shape[0]) - 1
    paged_kv_enabled = page_table is not None

    page_table_arr: mx.array
    if paged_kv_enabled:
        page_table_arr = mx.array(page_table, dtype=mx.int32)
        page_size_value, max_pages_value = validate_paged_kv_params(
            k_cache=k,
            v_cache=v,
            page_table=page_table_arr,
            batch_size=batch_size,
        )
        log2_page_size = log2_if_power_of_two(page_size_value)
        k_tensor = mx.reshape(k, (-1, k.shape[2], k.shape[3]))
        v_tensor = mx.reshape(v, (-1, v.shape[2], v.shape[3]))
    else:
        page_table_arr = _VARLEN_DUMMY_PAGE_TABLE
        page_size_value = 1
        max_pages_value = 1
        log2_page_size = -1
        k_tensor = k
        v_tensor = v

    params: VarlenAttentionParams = create_varlen_params(
        q=q,
        k=k_tensor,
        v=v_tensor,
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

    if params.total_q == 0 or params.batch_size == 0:
        empty_out = mx.zeros((0, params.nheads, params.headdim), dtype=q.dtype)
        empty_lse = mx.zeros((params.nheads, 0), dtype=mx.float32)
        return empty_out, empty_lse

    if not use_metal_kernel:
        out_ref, lse_ref, _ = varlen_attention_ref_mlx(
            q,
            k_tensor,
            v_tensor,
            cu_seqlens_q,
            cu_seqlens_k,
            dropout_p=dropout_p,
            softmax_scale=params.softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
        )
        return out_ref, lse_ref

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

    kernel = _get_varlen_forward_kernel()
    causal_int = 1 if causal else 0
    total_threads = params.total_q * params.nheads

    threadgroup_size = 128
    if total_threads < threadgroup_size:
        threadgroup_size = 32 if total_threads <= 32 else 64

    try:
        outputs = kernel(
            inputs=[
                q,
                k_tensor,
                v_tensor,
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
                params.window_size_left,
                params.window_size_right,
                alibi_slopes_arr,
                use_alibi,
                alibi_batch_stride,
                page_table_arr,
                page_size_value,
                max_pages_value,
                1 if paged_kv_enabled else 0,
                log2_page_size,
                dropout_p,
                philox_seed,
                philox_offset,
            ],
            template=[("T", q.dtype)],
            grid=(total_threads, 1, 1),
            threadgroup=(threadgroup_size, 1, 1),
            output_shapes=[
                (params.total_q, params.nheads, params.headdim),
                (params.nheads, params.total_q),
            ],
            output_dtypes=[q.dtype, mx.float32],
        )
        return outputs[0], outputs[1]
    except Exception:
        if paged_kv_enabled:
            raise
        out_ref, lse_ref, _ = varlen_attention_ref_mlx(
            q,
            k_tensor,
            v_tensor,
            cu_seqlens_q,
            cu_seqlens_k,
            dropout_p=dropout_p,
            softmax_scale=params.softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
        )
        return out_ref, lse_ref

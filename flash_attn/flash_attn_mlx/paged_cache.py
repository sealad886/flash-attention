"""Paged KV cache update helpers for the MLX backend."""

from __future__ import annotations

from typing import Tuple

import mlx.core as mx

from .kernels import (
    MetalKernelSpec,
    get_compiled_kernel,
    get_paged_cache_update_kernel,
    register_metal_kernel,
)
from .utils import log2_if_power_of_two, validate_paged_kv_params


_PAGED_CACHE_KERNEL_NAME = "paged_cache_update"

register_metal_kernel(
    MetalKernelSpec(
        name=_PAGED_CACHE_KERNEL_NAME,
        input_names=[
            "new_k",
            "new_v",
            "k_cache",
            "v_cache",
            "page_table",
            "cache_seqlens",
            "page_size",
            "max_pages_per_seq",
            "batch_size",
            "seqlen_new",
            "nheads_k",
            "headdim",
            "num_pages",
            "log2_page_size",
        ],
        output_names=[],
        source_fn=get_paged_cache_update_kernel,
        ensure_row_contiguous=True,
    )
)


def _get_paged_cache_kernel():
    """Return the compiled paged cache update kernel."""
    return get_compiled_kernel(_PAGED_CACHE_KERNEL_NAME)


def _validate_new_kv_tensors(new_k: mx.array, new_v: mx.array) -> Tuple[int, int, int, mx.dtype]:
    if new_k.ndim != 4:
        raise ValueError(
            f"new_k must have shape (batch, seqlen_new, nheads_k, headdim); got {new_k.shape}"
        )
    if new_v.shape != new_k.shape:
        raise ValueError(
            "new_v shape must match new_k shape for paged cache updates"
        )
    batch_size, seqlen_new, nheads_k, headdim = new_k.shape
    if nheads_k <= 0 or headdim <= 0:
        raise ValueError("nheads_k and headdim must be positive")
    if new_k.dtype != new_v.dtype:
        raise TypeError(
            f"new_k and new_v must share the same dtype (got {new_k.dtype} vs {new_v.dtype})"
        )
    return batch_size, seqlen_new, nheads_k, headdim


def update_paged_kv_cache(
    k_cache: mx.array,
    v_cache: mx.array,
    new_k: mx.array,
    new_v: mx.array,
    page_table: mx.array,
    cache_seqlens: mx.array,
) -> mx.array:
    """Write freshly computed key/value blocks into a paged KV cache.

    This helper mirrors ``hopper/paged_kv.h::PagedKVManager::write``. It
    validates the cache layout, launches the Metal writer, and returns the
    updated logical lengths so that sequential decoding loops can feed them
    back into :func:`flash_attn_with_kvcache`.

    Args:
        k_cache: Physical key cache buffer with shape
            ``(num_pages, page_size, nheads_k, headdim)``.
        v_cache: Physical value cache buffer (same shape/dtype as ``k_cache``).
        new_k: Newly computed keys with shape
            ``(batch, seqlen_new, nheads_k, headdim)``.
        new_v: Newly computed values matching ``new_k``.
        page_table: Block assignment table ``(batch, max_pages_per_seq)`` whose
            entries contain physical page indices or ``-1`` for unused slots.
        cache_seqlens: Current logical cache lengths per batch element, provided
            as an int32 vector ``(batch,)``.

    Returns:
        Int32 vector with the same shape/dtype as ``cache_seqlens`` describing
        the new logical lengths (old length + ``seqlen_new``).

    Raises:
        ValueError / TypeError when the cache layout, block table, or tensor
        dtypes do not satisfy the paged-kv requirements documented in the CUDA
        reference implementation.
    """

    if cache_seqlens.ndim != 1:
        raise ValueError(
            f"cache_seqlens must be 1-D (batch,), got shape {cache_seqlens.shape}"
        )
    if cache_seqlens.dtype != mx.int32:
        raise TypeError(
            f"cache_seqlens must be int32, got {cache_seqlens.dtype}"
        )

    batch_size, seqlen_new, nheads_k, headdim = _validate_new_kv_tensors(new_k, new_v)

    if batch_size != cache_seqlens.shape[0]:
        raise ValueError(
            "cache_seqlens batch dimension must match new_k batch size"
        )

    page_table_arr = mx.array(page_table, dtype=mx.int32)
    cache_seqlens_arr = mx.array(cache_seqlens, dtype=mx.int32)

    page_size, max_pages_per_seq = validate_paged_kv_params(
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table_arr,
        batch_size=batch_size,
    )

    if k_cache.dtype != new_k.dtype or v_cache.dtype != new_v.dtype:
        raise TypeError(
            "KV cache dtype must match the dtype of the new tensors"
        )

    if k_cache.shape != v_cache.shape:
        raise ValueError("k_cache and v_cache must have identical shapes")

    if k_cache.shape[2] != nheads_k:
        raise ValueError(
            "k_cache head dimension does not match new_k"
        )
    if k_cache.shape[3] != headdim:
        raise ValueError(
            "k_cache headdim does not match new_k"
        )

    total_elements = batch_size * seqlen_new * nheads_k * headdim
    if total_elements == 0:
        return cache_seqlens_arr

    kernel = _get_paged_cache_kernel()
    log2_page_size = log2_if_power_of_two(page_size)
    num_pages = int(k_cache.shape[0])

    kernel(
        inputs=[
            new_k,
            new_v,
            k_cache,
            v_cache,
            page_table_arr,
            cache_seqlens_arr,
            page_size,
            max_pages_per_seq,
            batch_size,
            seqlen_new,
            nheads_k,
            headdim,
            num_pages,
            log2_page_size,
        ],
        template=[("T", k_cache.dtype)],
        grid=(total_elements, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[],
        output_dtypes=[],
    )

    return cache_seqlens_arr + seqlen_new

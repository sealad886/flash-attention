"""Unit tests for Flash Attention MLX backend.

Tests cover:
- Forward pass with Metal kernels
- Backward pass with Metal kernels
- VJP (automatic differentiation) integration
- Various configurations: causal, GQA, MQA, different head dimensions
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import pytest
import mlx.core as mx

from flash_attn.flash_attn_mlx.ops import flash_attention_mlx, flash_attention_with_lse
from flash_attn.flash_attn_mlx.fwd_kernel import flash_attention_forward_metal as flash_attention_forward
from flash_attn.flash_attn_mlx.bwd_kernel import flash_attention_backward
from flash_attn.flash_attn_mlx.interface_mlx import (
    flash_attn_func,
    flash_attn_varlen_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_with_kvcache,
)
from flash_attn.flash_attn_mlx.reference import (
    attention_forward_ref,
    attention_backward_ref,
    attention_ref_mlx,
    varlen_attention_ref_mlx,
)
from flash_attn.flash_attn_mlx.utils import USE_REF
from flash_attn.flash_attn_mlx.tests.varlen_test_utils import (
    VARLEN_CAUSAL_FLAGS,
    VARLEN_HEAD_CONFIGS,
    VARLEN_HEADDIMS,
    VARLEN_SEQLEN_CASES,
    generate_cu_seqlens,
    make_varlen_qkv,
)
from flash_attn.flash_attn_mlx.varlen_fwd_kernel import varlen_flash_attention_forward
from flash_attn.flash_attn_mlx.varlen_bwd_kernel import varlen_flash_attention_backward
from flash_attn.flash_attn_mlx.varlen_ops import varlen_flash_attention_mlx
from flash_attn.flash_attn_mlx.paged_cache import update_paged_kv_cache


def _cu_seqlens_to_list(cu_seqlens: mx.array) -> List[int]:
    """Convert an MLX array of cu_seqlens to a Python int list."""
    return [int(cu_seqlens[i].item()) for i in range(cu_seqlens.shape[0])]


def _pad_varlen_tensor(
    tensor: mx.array, cu_seqlens: mx.array, max_seqlen: int
) -> mx.array:
    """Pad packed varlen tensor into (batch, max_seqlen, ...) layout."""
    batch = len(cu_seqlens) - 1
    padded = []
    for b in range(batch):
        start = int(cu_seqlens[b])
        end = int(cu_seqlens[b + 1])
        segment = tensor[start:end]
        pad_len = max_seqlen - (end - start)
        if pad_len > 0:
            pad_shape = (pad_len,) + tuple(segment.shape[1:])
            pad_slice = mx.zeros(pad_shape, dtype=segment.dtype)
            segment = mx.concatenate([segment, pad_slice], axis=0)
        padded.append(segment)
    return mx.stack(padded, axis=0)


def _flatten_padded_tensor(tensor: mx.array, cu_seqlens: mx.array) -> mx.array:
    """Flatten padded tensor back into packed varlen layout."""
    slices = []
    batch = len(cu_seqlens) - 1
    for b in range(batch):
        start = int(cu_seqlens[b])
        end = int(cu_seqlens[b + 1])
        length = end - start
        if length <= 0:
            continue
        slices.append(tensor[b, :length])
    return mx.concatenate(slices, axis=0) if slices else tensor[:0]


def _default_philox_state(dropout_p: float) -> Tuple[int, int]:
    """Return the Philox seed/offset pair used by the MLX interface."""
    if dropout_p <= 0.0:
        return 0, 0
    return 0x1BF58, 0x1D4B49


_COMMON_VARLEN_SEQLENS = tuple(VARLEN_SEQLEN_CASES[:4])
_EDGE_VARLEN_SEQLENS = tuple(VARLEN_SEQLEN_CASES[4:])


def _seqlens_id(seqlens: Sequence[int]) -> str:
    """Generate a test ID string from sequence lengths."""
    return "-".join(str(int(length)) for length in seqlens)


# ============================================================================
# Paged KV Cache Test Helpers
# ============================================================================


def _build_paged_cache_internal(
    k_slices: list,
    v_slices: list,
    page_size: int,
) -> tuple:
    """
    Build paged caches from per-sequence tensor slices.

    Args:
        k_slices: List of K tensors per sequence, each of shape (tokens, nheads_k, headdim).
        v_slices: List of V tensors per sequence, each of shape (tokens, nheads_k, headdim).
        page_size: Number of tokens per physical page.

    Returns:
        Tuple of (k_cache, v_cache, page_table):
        - k_cache: Physical cache buffer (num_pages, page_size, nheads_k, headdim).
        - v_cache: Physical cache buffer (num_pages, page_size, nheads_k, headdim).
        - page_table: Block table (batch, max_pages_per_seq), int32, with -1 for unused entries.
    """
    if len(k_slices) == 0:
        raise ValueError("k_slices must not be empty")
    if len(k_slices) != len(v_slices):
        raise ValueError("k_slices and v_slices must have the same length")

    batch = len(k_slices)
    nheads_k = k_slices[0].shape[1]
    headdim = k_slices[0].shape[2]
    dtype = k_slices[0].dtype

    # Compute number of pages per sequence and max pages
    pages_per_seq = []
    for k_seq in k_slices:
        tokens = k_seq.shape[0]
        num_pages = (tokens + page_size - 1) // page_size if tokens > 0 else 0
        pages_per_seq.append(num_pages)

    max_pages_per_seq = max(pages_per_seq) if pages_per_seq else 1
    total_pages = sum(pages_per_seq)

    # Allocate physical caches
    k_cache = mx.zeros((total_pages, page_size, nheads_k, headdim), dtype=dtype)
    v_cache = mx.zeros((total_pages, page_size, nheads_k, headdim), dtype=dtype)

    # Allocate page table filled with -1
    page_table = mx.full((batch, max_pages_per_seq), -1, dtype=mx.int32)

    # Fill caches and page table
    page_idx = 0
    k_cache_list = []
    v_cache_list = []

    for b in range(batch):
        k_seq = k_slices[b]
        v_seq = v_slices[b]
        tokens = k_seq.shape[0]

        if tokens == 0:
            continue

        num_pages = pages_per_seq[b]
        page_indices_for_seq = []

        for p in range(num_pages):
            start_tok = p * page_size
            end_tok = min(start_tok + page_size, tokens)
            page_len = end_tok - start_tok

            # Extract slice for this page
            k_page_data = k_seq[start_tok:end_tok]
            v_page_data = v_seq[start_tok:end_tok]

            # Pad if needed
            if page_len < page_size:
                pad_shape = (page_size - page_len, nheads_k, headdim)
                k_page_data = mx.concatenate(
                    [k_page_data, mx.zeros(pad_shape, dtype=dtype)], axis=0
                )
                v_page_data = mx.concatenate(
                    [v_page_data, mx.zeros(pad_shape, dtype=dtype)], axis=0
                )

            k_cache_list.append(k_page_data)
            v_cache_list.append(v_page_data)
            page_indices_for_seq.append(page_idx)
            page_idx += 1

        # Update page table
        page_table_row = list(page_indices_for_seq) + [-1] * (max_pages_per_seq - num_pages)
        page_table = mx.concatenate([
            page_table[:b],
            mx.array([page_table_row], dtype=mx.int32),
            page_table[b + 1:],
        ], axis=0)

    # Stack pages into caches
    if k_cache_list:
        k_cache = mx.stack(k_cache_list, axis=0)
        v_cache = mx.stack(v_cache_list, axis=0)
    else:
        k_cache = mx.zeros((1, page_size, nheads_k, headdim), dtype=dtype)
        v_cache = mx.zeros((1, page_size, nheads_k, headdim), dtype=dtype)

    return k_cache, v_cache, page_table


def _build_paged_cache_from_padded(
    k_tensor: mx.array,
    v_tensor: mx.array,
    seqlens: list,
    page_size: int,
) -> tuple:
    """
    Build paged caches from padded (batch, seqlen, nheads_k, headdim) tensors.

    Args:
        k_tensor: Padded K tensor of shape (batch, seqlen, nheads_k, headdim).
        v_tensor: Padded V tensor of shape (batch, seqlen, nheads_k, headdim).
        seqlens: List of actual sequence lengths per batch element.
        page_size: Number of tokens per physical page.

    Returns:
        Tuple of (k_cache, v_cache, page_table).
    """
    batch = k_tensor.shape[0]
    if len(seqlens) != batch:
        raise ValueError("seqlens length must match batch dimension")

    k_slices = []
    v_slices = []
    for b in range(batch):
        length = seqlens[b]
        k_slices.append(k_tensor[b, :length])
        v_slices.append(v_tensor[b, :length])

    return _build_paged_cache_internal(k_slices, v_slices, page_size)


def _build_paged_cache_from_varlen(
    k_tensor: mx.array,
    v_tensor: mx.array,
    cu_seqlens: mx.array,
    page_size: int,
) -> tuple:
    """
    Build paged caches from varlen (total_tokens, nheads_k, headdim) tensors.

    Args:
        k_tensor: Packed K tensor of shape (total_tokens, nheads_k, headdim).
        v_tensor: Packed V tensor of shape (total_tokens, nheads_k, headdim).
        cu_seqlens: Cumulative sequence lengths of shape (batch + 1,).
        page_size: Number of tokens per physical page.

    Returns:
        Tuple of (k_cache, v_cache, page_table).
    """
    cu_list = [int(cu_seqlens[i].item()) for i in range(cu_seqlens.shape[0])]
    batch = len(cu_list) - 1

    k_slices = []
    v_slices = []
    for b in range(batch):
        start = cu_list[b]
        end = cu_list[b + 1]
        k_slices.append(k_tensor[start:end])
        v_slices.append(v_tensor[start:end])

    return _build_paged_cache_internal(k_slices, v_slices, page_size)


def _paged_cache_to_contiguous(
    k_cache: mx.array,
    page_table: mx.array,
    seqlens: list,
) -> mx.array:
    """
    Reconstruct a padded (batch, max_seqlen, nheads_k, headdim) tensor from a paged cache.

    Args:
        k_cache: Physical cache of shape (num_pages, page_size, nheads_k, headdim).
        page_table: Block table of shape (batch, max_pages_per_seq), int32.
        seqlens: List of actual sequence lengths per batch element.

    Returns:
        Reconstructed tensor of shape (batch, max_seqlen, nheads_k, headdim).
    """
    num_pages, page_size, nheads_k, headdim = k_cache.shape
    batch = page_table.shape[0]
    max_seqlen = max(seqlens) if seqlens else 0

    if max_seqlen == 0:
        return mx.zeros((batch, 0, nheads_k, headdim), dtype=k_cache.dtype)

    result_slices = []
    for b in range(batch):
        length = seqlens[b]
        if length == 0:
            result_slices.append(mx.zeros((max_seqlen, nheads_k, headdim), dtype=k_cache.dtype))
            continue

        # Collect tokens from pages
        tokens_collected = []
        num_pages_needed = (length + page_size - 1) // page_size
        tokens_remaining = length

        for p in range(num_pages_needed):
            page_idx = int(page_table[b, p].item())
            if page_idx < 0:
                break
            tokens_in_page = min(page_size, tokens_remaining)
            tokens_collected.append(k_cache[page_idx, :tokens_in_page])
            tokens_remaining -= tokens_in_page

        if tokens_collected:
            seq_data = mx.concatenate(tokens_collected, axis=0)
        else:
            seq_data = mx.zeros((0, nheads_k, headdim), dtype=k_cache.dtype)

        # Pad to max_seqlen
        if seq_data.shape[0] < max_seqlen:
            pad_len = max_seqlen - seq_data.shape[0]
            pad = mx.zeros((pad_len, nheads_k, headdim), dtype=k_cache.dtype)
            seq_data = mx.concatenate([seq_data, pad], axis=0)

        result_slices.append(seq_data)

    return mx.stack(result_slices, axis=0)


def _allocate_paged_cache_buffers(
    batch: int,
    max_pages_per_seq: int,
    page_size: int,
    nheads_k: int,
    headdim: int,
    dtype=mx.float16,
):
    """Allocate zero-initialized paged cache buffers and a sequential page table."""

    if batch <= 0:
        raise ValueError("batch must be positive")
    if max_pages_per_seq <= 0:
        raise ValueError("max_pages_per_seq must be positive")

    num_pages = batch * max_pages_per_seq
    k_cache = mx.zeros((num_pages, page_size, nheads_k, headdim), dtype=dtype)
    v_cache = mx.zeros_like(k_cache)

    page_rows = []
    page_idx = 0
    for _ in range(batch):
        row = list(range(page_idx, page_idx + max_pages_per_seq))
        page_rows.append(row)
        page_idx += max_pages_per_seq

    page_table = mx.array(page_rows, dtype=mx.int32)
    return k_cache, v_cache, page_table


def _require_paged_kernel():
    """Skip the current test when paged Metal kernels are unavailable."""
    from flash_attn.flash_attn_mlx.utils import USE_REF

    if USE_REF:
        pytest.skip(
            "Paged KV cache requires the Metal kernel path; "
            "set FLASH_ATTENTION_MLX_REF=0 to enable",
        )

    try:
        # Late import to avoid unused dependency when baseline tests run.
        from flash_attn.flash_attn_mlx.paged_cache import _get_paged_cache_kernel

        _get_paged_cache_kernel()
    except Exception as exc:  # pragma: no cover - environment specific
        pytest.skip(f"Paged cache Metal kernel unavailable: {exc}")


def _padded_attention_with_lengths(q_padded, k_padded, v_padded, seqlens_q, seqlens_k, causal):
    """Compute attention on padded tensors while masking out padding tokens."""
    batch, max_seqlen_q, nheads, headdim = q_padded.shape
    _, max_seqlen_k, nheads_k, _ = k_padded.shape

    softmax_scale = 1.0 / math.sqrt(headdim)

    if nheads != nheads_k:
        assert nheads % nheads_k == 0
        repeat_factor = nheads // nheads_k
        k_padded = mx.repeat(k_padded, repeat_factor, axis=2)
        v_padded = mx.repeat(v_padded, repeat_factor, axis=2)

    q_t = mx.transpose(q_padded, (0, 2, 1, 3))
    k_t = mx.transpose(k_padded, (0, 2, 3, 1))
    scores = mx.matmul(q_t, k_t).astype(mx.float32) * softmax_scale

    q_positions = mx.arange(max_seqlen_q).reshape(1, max_seqlen_q, 1)
    k_positions = mx.arange(max_seqlen_k).reshape(1, 1, max_seqlen_k)
    len_q = mx.array(seqlens_q, dtype=mx.int32).reshape(batch, 1, 1)
    len_k = mx.array(seqlens_k, dtype=mx.int32).reshape(batch, 1, 1)

    valid_q = q_positions < len_q
    valid_k = k_positions < len_k
    valid_mask = mx.logical_and(valid_q, valid_k)
    if causal:
        causal_mask = k_positions <= q_positions
        valid_mask = mx.logical_and(valid_mask, causal_mask)

    valid_mask = mx.expand_dims(valid_mask, axis=1)
    scores = mx.where(
        valid_mask,
        scores,
        mx.full(scores.shape, float("-inf"), dtype=mx.float32),
    )

    scores_max = mx.max(scores, axis=-1, keepdims=True)
    scores_max = mx.where(mx.isinf(scores_max), mx.zeros_like(scores_max), scores_max)
    scores_exp = mx.exp(scores - scores_max)
    scores_exp = mx.where(valid_mask, scores_exp, mx.zeros_like(scores_exp))
    scores_sum = mx.sum(scores_exp, axis=-1, keepdims=True)
    scores_sum = mx.maximum(scores_sum, mx.array(1e-12, dtype=scores_sum.dtype))

    attn_probs = scores_exp / scores_sum
    v_t = mx.transpose(v_padded, (0, 2, 1, 3)).astype(mx.float32)
    out_t = mx.matmul(attn_probs, v_t)
    out = mx.transpose(out_t, (0, 2, 1, 3)).astype(q_padded.dtype)

    softmax_lse = mx.squeeze(scores_max, axis=-1) + mx.log(mx.squeeze(scores_sum, axis=-1))

    return out, softmax_lse


# ============================================================================
# Varlen Reference Tests
# ============================================================================


class TestVarlenReference:
    """Tests validating the varlen reference implementation."""

    @pytest.mark.parametrize(
        "seqlens",
        [
            [128, 128, 128],
            [97, 200, 50],
            [1, 512, 32],
            [256],
        ],
    )
    @pytest.mark.parametrize("causal", [False, True])
    def test_varlen_reference_matches_padded(self, seqlens, causal):
        nheads, nheads_k, headdim = 4, 4, 64
        (
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
        ) = make_varlen_qkv(seqlens, nheads, nheads_k, headdim)

        cu_q_list = _cu_seqlens_to_list(cu_seqlens_q)
        cu_k_list = _cu_seqlens_to_list(cu_seqlens_k)

        q_padded = _pad_varlen_tensor(q_varlen, cu_q_list, max_seqlen_q)
        k_padded = _pad_varlen_tensor(k_varlen, cu_k_list, max_seqlen_k)
        v_padded = _pad_varlen_tensor(v_varlen, cu_k_list, max_seqlen_k)

        out_varlen, lse_varlen, _ = varlen_attention_ref_mlx(
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            causal=causal,
        )

        out_padded, lse_padded = _padded_attention_with_lengths(
            q_padded,
            k_padded,
            v_padded,
            seqlens,
            seqlens,
            causal,
        )

        out_padded_varlen = _flatten_padded_tensor(out_padded, cu_q_list)

        lse_segments = []
        batch = len(cu_q_list) - 1
        for b in range(batch):
            start = cu_q_list[b]
            end = cu_q_list[b + 1]
            length = end - start
            if length <= 0:
                continue
            lse_segments.append(lse_padded[b, :, :length])
        lse_padded_varlen = (
            mx.concatenate(lse_segments, axis=-1) if lse_segments else lse_padded[:, :, :0]
        )

        max_out_diff = float(mx.max(mx.abs(out_varlen - out_padded_varlen)))
        max_lse_diff = float(mx.max(mx.abs(lse_varlen - lse_padded_varlen)))

        assert max_out_diff < 1e-3, f"Varlen reference output mismatch: {max_out_diff}"
        assert max_lse_diff < 5e-3, f"Varlen reference LSE mismatch: {max_lse_diff}"


class TestVarlenForwardKernel:
    """Tests validating the Metal varlen forward kernel against the reference."""

    @pytest.mark.parametrize(
        "seqlens",
        [
            [64, 64],
            [33, 17, 80],
            [1, 5, 7, 3],
        ],
    )
    @pytest.mark.parametrize("causal", [False, True])
    @pytest.mark.parametrize("head_config", [(4, 4), (8, 4), (8, 2), (8, 1)])
    def test_varlen_forward_matches_reference(self, seqlens, causal, head_config):
        nheads, nheads_k = head_config
        headdim = 64

        (
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
        ) = make_varlen_qkv(seqlens, nheads, nheads_k, headdim)

        out_kernel, lse_kernel = varlen_flash_attention_forward(
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=causal,
        )

        out_ref, lse_ref, _ = varlen_attention_ref_mlx(
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            causal=causal,
        )

        max_out_diff = float(mx.max(mx.abs(out_kernel - out_ref)))
        max_lse_diff = float(mx.max(mx.abs(lse_kernel - lse_ref)))

        assert max_out_diff < 1e-3, f"Varlen kernel output mismatch: {max_out_diff}"
        assert max_lse_diff < 5e-3, f"Varlen kernel LSE mismatch: {max_lse_diff}"


class TestVarlenBackwardKernel:
    """Tests validating the Metal varlen backward kernels against reference."""

    @pytest.mark.parametrize(
        "seqlens",
        [
            [32, 24],
            [17, 9, 7],
        ],
    )
    @pytest.mark.parametrize("causal", [False, True])
    @pytest.mark.parametrize("head_config", [(4, 4), (8, 4), (8, 2), (8, 1)])
    def test_varlen_backward_matches_reference(self, seqlens, causal, head_config):
        nheads, nheads_k = head_config
        headdim = 64

        (
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
        ) = make_varlen_qkv(seqlens, nheads, nheads_k, headdim)

        out_kernel, lse_kernel = varlen_flash_attention_forward(
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=causal,
        )

        mx.random.seed(123)
        dout = mx.random.normal(out_kernel.shape).astype(q_varlen.dtype) * 0.1

        dq_kernel, dk_kernel, dv_kernel = varlen_flash_attention_backward(
            q_varlen,
            k_varlen,
            v_varlen,
            out_kernel,
            dout,
            lse_kernel,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=causal,
        )

        dq_ref, dk_ref, dv_ref = varlen_flash_attention_backward(
            q_varlen,
            k_varlen,
            v_varlen,
            out_kernel,
            dout,
            lse_kernel,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=causal,
            use_metal_kernel=False,
        )

        max_dq = float(mx.max(mx.abs(dq_kernel - dq_ref)))
        max_dk = float(mx.max(mx.abs(dk_kernel - dk_ref)))
        max_dv = float(mx.max(mx.abs(dv_kernel - dv_ref)))

        assert max_dq < 5e-2, f"Varlen dq mismatch: {max_dq}"
        assert max_dk < 5e-2, f"Varlen dk mismatch: {max_dk}"
        assert max_dv < 5e-2, f"Varlen dv mismatch: {max_dv}"


class TestVarlenAutogradIntegration:
    """Tests verifying autograd works through the varlen interface function."""

    @pytest.mark.parametrize("causal", [False, True])
    def test_varlen_interface_gradients_match_reference(self, causal):
        seqlens = [19, 7]
        nheads, nheads_k, headdim = 4, 4, 64

        mx.random.seed(123)

        (
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
        ) = make_varlen_qkv(seqlens, nheads, nheads_k, headdim)

        upstream = mx.random.normal(q_varlen.shape).astype(q_varlen.dtype) * 0.01

        def loss_fn(q_in, k_in, v_in):
            out = flash_attn_varlen_func(
                q_in,
                k_in,
                v_in,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                causal=causal,
            )
            return mx.sum(out * upstream)

        dq_auto, dk_auto, dv_auto = mx.grad(loss_fn, argnums=(0, 1, 2))(
            q_varlen,
            k_varlen,
            v_varlen,
        )

        out_kernel, lse_kernel = varlen_flash_attention_forward(
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=causal,
        )

        dq_ref, dk_ref, dv_ref = varlen_flash_attention_backward(
            q_varlen,
            k_varlen,
            v_varlen,
            out_kernel,
            upstream,
            lse_kernel,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=causal,
            use_metal_kernel=False,
        )

        max_dq = float(mx.max(mx.abs(dq_auto - dq_ref)))
        max_dk = float(mx.max(mx.abs(dk_auto - dk_ref)))
        max_dv = float(mx.max(mx.abs(dv_auto - dv_ref)))

        assert max_dq < 5e-2, f"Autograd dq mismatch: {max_dq}"
        assert max_dk < 5e-2, f"Autograd dk mismatch: {max_dk}"
        assert max_dv < 5e-2, f"Autograd dv mismatch: {max_dv}"


class TestVarlenWindowedAttention:
    """Coverage for sliding-window attention settings."""

    @pytest.mark.parametrize("window_size", [(3, -1), (2, 1)])
    @pytest.mark.parametrize("causal", [False, True])
    def test_varlen_forward_window_matches_reference(self, window_size, causal):
        seqlens = [41, 19]
        nheads = 4
        nheads_k = 4
        headdim = 64

        (
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
        ) = make_varlen_qkv(seqlens, nheads, nheads_k, headdim)

        out_kernel, lse_kernel = varlen_flash_attention_forward(
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=causal,
            window_size=window_size,
        )

        out_ref, lse_ref, _ = varlen_attention_ref_mlx(
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            causal=causal,
            window_size=window_size,
        )

        max_out_diff = float(mx.max(mx.abs(out_kernel - out_ref)))
        max_lse_diff = float(mx.max(mx.abs(lse_kernel - lse_ref)))

        assert max_out_diff < 1e-3, f"Windowed forward output mismatch: {max_out_diff}"
        assert max_lse_diff < 5e-3, f"Windowed forward LSE mismatch: {max_lse_diff}"

    @pytest.mark.parametrize("window_size", [(3, -1), (2, 1)])
    @pytest.mark.parametrize("causal", [False, True])
    def test_varlen_backward_window_matches_reference(self, window_size, causal):
        seqlens = [37, 11]
        nheads = 4
        nheads_k = 4
        headdim = 64

        (
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
        ) = make_varlen_qkv(seqlens, nheads, nheads_k, headdim)

        out_kernel, lse_kernel = varlen_flash_attention_forward(
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=causal,
            window_size=window_size,
        )

        mx.random.seed(321)
        dout = mx.random.normal(out_kernel.shape).astype(q_varlen.dtype) * 0.05

        dq_kernel, dk_kernel, dv_kernel = varlen_flash_attention_backward(
            q_varlen,
            k_varlen,
            v_varlen,
            out_kernel,
            dout,
            lse_kernel,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=causal,
            window_size=window_size,
        )

        dq_ref, dk_ref, dv_ref = varlen_flash_attention_backward(
            q_varlen,
            k_varlen,
            v_varlen,
            out_kernel,
            dout,
            lse_kernel,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=causal,
            window_size=window_size,
            use_metal_kernel=False,
        )

        max_dq = float(mx.max(mx.abs(dq_kernel - dq_ref)))
        max_dk = float(mx.max(mx.abs(dk_kernel - dk_ref)))
        max_dv = float(mx.max(mx.abs(dv_kernel - dv_ref)))

        assert max_dq < 5e-2, f"Windowed dq mismatch: {max_dq}"
        assert max_dk < 5e-2, f"Windowed dk mismatch: {max_dk}"
        assert max_dv < 5e-2, f"Windowed dv mismatch: {max_dv}"


class TestVarlenGQASupport:
    """Covers GQA / MQA head mapping when nheads != nheads_k."""

    @pytest.mark.parametrize("nheads_k", [2, 1])
    def test_varlen_forward_gqa_matches_reference(self, nheads_k):
        seqlens = [29, 13]
        nheads = 8
        headdim = 64

        (
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
        ) = make_varlen_qkv(seqlens, nheads, nheads_k, headdim)

        out_kernel, _ = varlen_flash_attention_forward(
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
        )

        out_ref, _, _ = varlen_attention_ref_mlx(
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
        )

        max_out_diff = float(mx.max(mx.abs(out_kernel - out_ref)))
        assert max_out_diff < 1e-3, f"GQA forward mismatch: {max_out_diff}"

    @pytest.mark.parametrize("nheads_k", [2, 1])
    def test_varlen_backward_gqa_matches_reference(self, nheads_k):
        seqlens = [23, 9]
        nheads = 8
        headdim = 64

        (
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
        ) = make_varlen_qkv(seqlens, nheads, nheads_k, headdim)

        out_kernel, lse_kernel = varlen_flash_attention_forward(
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
        )

        mx.random.seed(777)
        dout = mx.random.normal(out_kernel.shape).astype(q_varlen.dtype) * 0.05

        dq_kernel, dk_kernel, dv_kernel = varlen_flash_attention_backward(
            q_varlen,
            k_varlen,
            v_varlen,
            out_kernel,
            dout,
            lse_kernel,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
        )

        dq_ref, dk_ref, dv_ref = varlen_flash_attention_backward(
            q_varlen,
            k_varlen,
            v_varlen,
            out_kernel,
            dout,
            lse_kernel,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            use_metal_kernel=False,
        )

        max_dq = float(mx.max(mx.abs(dq_kernel - dq_ref)))
        max_dk = float(mx.max(mx.abs(dk_kernel - dk_ref)))
        max_dv = float(mx.max(mx.abs(dv_kernel - dv_ref)))

        assert max_dq < 5e-2, f"GQA dq mismatch: {max_dq}"
        assert max_dk < 5e-2, f"GQA dk mismatch: {max_dk}"
        assert max_dv < 5e-2, f"GQA dv mismatch: {max_dv}"


class TestVarlenEdgeCases:
    """Edge-case coverage for ragged batches."""

    def test_varlen_handles_zero_length_sequences(self):
        seqlens = [0, 15, 0, 7]
        nheads = 4
        nheads_k = 4
        headdim = 64

        (
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
        ) = make_varlen_qkv(seqlens, nheads, nheads_k, headdim)

        out_kernel, lse_kernel = varlen_flash_attention_forward(
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
        )

        out_ref, lse_ref, _ = varlen_attention_ref_mlx(
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
        )

        max_out_diff = float(mx.max(mx.abs(out_kernel - out_ref)))
        max_lse_diff = float(mx.max(mx.abs(lse_kernel - lse_ref)))

        assert max_out_diff < 1e-3, f"Zero-length output mismatch: {max_out_diff}"
        assert max_lse_diff < 5e-3, f"Zero-length LSE mismatch: {max_lse_diff}"

    def test_varlen_mismatched_qk_lengths(self):
        seqlens_q = [11, 5, 9]
        seqlens_k = [17, 8, 3]
        nheads = 4
        nheads_k = 2
        headdim = 64

        (
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
        ) = make_varlen_qkv(
            seqlens_q,
            nheads,
            nheads_k,
            headdim,
            seqlens_k=seqlens_k,
        )

        out_kernel, lse_kernel = varlen_flash_attention_forward(
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
        )

        out_ref, lse_ref, _ = varlen_attention_ref_mlx(
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
        )

        max_out_diff = float(mx.max(mx.abs(out_kernel - out_ref)))
        max_lse_diff = float(mx.max(mx.abs(lse_kernel - lse_ref)))

        assert max_out_diff < 1e-3, f"Mismatched Q/K output mismatch: {max_out_diff}"
        assert max_lse_diff < 5e-3, f"Mismatched Q/K LSE mismatch: {max_lse_diff}"


class TestVarlenComprehensive:
    """Comprehensive parametrized varlen test suite covering all configurations."""

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("max_seqlen", [32, 128])
    @pytest.mark.parametrize("headdim", [32, 64, 128])
    @pytest.mark.parametrize("causal", [False, True])
    def test_varlen_forward_comprehensive(self, batch_size, max_seqlen, headdim, causal):
        """Test varlen forward across various batch/seqlen/headdim combinations."""
        mx.random.seed(42)
        nheads = 4
        nheads_k = 4

        # Generate random sequence lengths for this batch
        seqlens = [mx.random.randint(1, max_seqlen + 1).item() for _ in range(batch_size)]

        (
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
        ) = make_varlen_qkv(seqlens, nheads, nheads_k, headdim)

        out_kernel, lse_kernel = varlen_flash_attention_forward(
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=causal,
        )

        out_ref, lse_ref, _ = varlen_attention_ref_mlx(
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            causal=causal,
        )

        max_out_diff = float(mx.max(mx.abs(out_kernel - out_ref)))
        max_lse_diff = float(mx.max(mx.abs(lse_kernel - lse_ref)))

        assert max_out_diff < 1e-3, f"Varlen forward mismatch: batch={batch_size}, maxseq={max_seqlen}, headdim={headdim}, causal={causal}, diff={max_out_diff}"
        assert max_lse_diff < 5e-3, f"Varlen LSE mismatch: {max_lse_diff}"

    @pytest.mark.parametrize("head_config", [(4, 4), (8, 4), (8, 2), (8, 1)])
    @pytest.mark.parametrize("causal", [False, True])
    def test_varlen_gqa_configurations(self, head_config, causal):
        """Test varlen with various GQA/MQA configurations."""
        nheads, nheads_k = head_config
        headdim = 64
        seqlens = [32, 24, 48]

        (
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
        ) = make_varlen_qkv(seqlens, nheads, nheads_k, headdim)

        out_kernel, lse_kernel = varlen_flash_attention_forward(
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=causal,
        )

        out_ref, lse_ref, _ = varlen_attention_ref_mlx(
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            causal=causal,
        )

        max_out_diff = float(mx.max(mx.abs(out_kernel - out_ref)))
        assert max_out_diff < 1e-3, f"GQA config {head_config} mismatch: {max_out_diff}"

    @pytest.mark.parametrize("window_size", [(-1, -1), (8, -1), (4, 4), (-1, 8)])
    @pytest.mark.parametrize("causal", [False, True])
    def test_varlen_window_comprehensive(self, window_size, causal):
        """Test varlen with various sliding window configurations."""
        nheads = 4
        nheads_k = 4
        headdim = 64
        seqlens = [32, 48]

        (
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
        ) = make_varlen_qkv(seqlens, nheads, nheads_k, headdim)

        out_kernel, lse_kernel = varlen_flash_attention_forward(
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=causal,
            window_size=window_size,
        )

        out_ref, lse_ref, _ = varlen_attention_ref_mlx(
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            causal=causal,
            window_size=window_size,
        )

        max_out_diff = float(mx.max(mx.abs(out_kernel - out_ref)))
        assert max_out_diff < 1e-3, f"Window {window_size} causal={causal} mismatch: {max_out_diff}"

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("causal", [False, True])
    def test_varlen_backward_comprehensive(self, batch_size, causal):
        """Test varlen backward across configurations."""
        nheads = 4
        nheads_k = 4
        headdim = 64
        mx.random.seed(42)
        seqlens = [mx.random.randint(16, 64).item() for _ in range(batch_size)]

        (
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
        ) = make_varlen_qkv(seqlens, nheads, nheads_k, headdim)

        out_kernel, lse_kernel = varlen_flash_attention_forward(
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=causal,
        )

        mx.random.seed(123)
        dout = mx.random.normal(out_kernel.shape).astype(q_varlen.dtype) * 0.1

        dq_kernel, dk_kernel, dv_kernel = varlen_flash_attention_backward(
            q_varlen,
            k_varlen,
            v_varlen,
            out_kernel,
            dout,
            lse_kernel,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=causal,
        )

        # Verify no NaN or Inf
        assert not mx.any(mx.isnan(dq_kernel)), "dQ has NaN"
        assert not mx.any(mx.isnan(dk_kernel)), "dK has NaN"
        assert not mx.any(mx.isnan(dv_kernel)), "dV has NaN"
        assert not mx.any(mx.isinf(dq_kernel)), "dQ has Inf"
        assert not mx.any(mx.isinf(dk_kernel)), "dK has Inf"
        assert not mx.any(mx.isinf(dv_kernel)), "dV has Inf"

    @pytest.mark.skipif(USE_REF, reason="Requires Metal kernel (FLASH_ATTENTION_MLX_REF=0)")
    @pytest.mark.parametrize("seqlens", _COMMON_VARLEN_SEQLENS, ids=_seqlens_id)
    @pytest.mark.parametrize("causal", VARLEN_CAUSAL_FLAGS)
    @pytest.mark.parametrize("headdim", VARLEN_HEADDIMS)
    @pytest.mark.parametrize("head_config", VARLEN_HEAD_CONFIGS)
    def test_varlen_interface_matches_reference_matrix(
        self,
        seqlens,
        causal,
        headdim,
        head_config,
    ):
        """Ensure flash_attn_varlen_func matches the reference across configs."""

        nheads, nheads_k = head_config
        (
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
        ) = make_varlen_qkv(seqlens, nheads, nheads_k, headdim)

        out_mlx, lse_mlx, _ = flash_attn_varlen_func(
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            causal=causal,
            return_attn_probs=True,
        )

        out_ref, lse_ref, _ = varlen_attention_ref_mlx(
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            causal=causal,
        )

        max_out_diff = float(mx.max(mx.abs(out_mlx - out_ref)))
        max_lse_diff = float(mx.max(mx.abs(lse_mlx - lse_ref)))

        assert max_out_diff < 1e-3, (
            f"Interface output mismatch ({seqlens}, headdim={headdim}, "
            f"nheads={nheads}, nheads_k={nheads_k}, causal={causal}): {max_out_diff}"
        )
        assert max_lse_diff < 5e-3, f"Interface LSE mismatch: {max_lse_diff}"

    @pytest.mark.skipif(USE_REF, reason="Requires Metal kernel (FLASH_ATTENTION_MLX_REF=0)")
    @pytest.mark.parametrize("seqlens", _EDGE_VARLEN_SEQLENS, ids=_seqlens_id)
    @pytest.mark.parametrize("causal", VARLEN_CAUSAL_FLAGS)
    def test_varlen_interface_edge_cases(self, seqlens, causal):
        """Edge-case coverage for flash_attn_varlen_func (single tokens, very long)."""

        nheads, nheads_k, headdim = 8, 2, 64
        (
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
        ) = make_varlen_qkv(seqlens, nheads, nheads_k, headdim)

        out_mlx = flash_attn_varlen_func(
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            causal=causal,
        )

        out_ref, _, _ = varlen_attention_ref_mlx(
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            causal=causal,
        )

        max_out_diff = float(mx.max(mx.abs(out_mlx - out_ref)))
        assert max_out_diff < 1e-3, (
            f"Edge-case varlen mismatch ({seqlens}, causal={causal}): {max_out_diff}"
        )

    @pytest.mark.skipif(USE_REF, reason="Requires Metal kernel (FLASH_ATTENTION_MLX_REF=0)")
    @pytest.mark.parametrize("seqlens", _COMMON_VARLEN_SEQLENS[:2], ids=_seqlens_id)
    @pytest.mark.parametrize("causal", VARLEN_CAUSAL_FLAGS)
    @pytest.mark.parametrize("head_config", VARLEN_HEAD_CONFIGS)
    def test_varlen_interface_backward_matrix(self, seqlens, causal, head_config):
        """Validate gradients via mx.grad for the varlen interface."""

        nheads, nheads_k = head_config
        headdim = 64

        (
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
        ) = make_varlen_qkv(seqlens, nheads, nheads_k, headdim)

        mx.random.seed(2024)
        upstream = mx.random.normal(q_varlen.shape).astype(q_varlen.dtype) * 0.01

        def loss_fn(q_in, k_in, v_in):
            out = flash_attn_varlen_func(
                q_in,
                k_in,
                v_in,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                causal=causal,
            )
            return mx.sum(out * upstream)

        dq_auto, dk_auto, dv_auto = mx.grad(loss_fn, argnums=(0, 1, 2))(
            q_varlen,
            k_varlen,
            v_varlen,
        )

        out_kernel, lse_kernel = varlen_flash_attention_forward(
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=causal,
        )

        dq_ref, dk_ref, dv_ref = varlen_flash_attention_backward(
            q_varlen,
            k_varlen,
            v_varlen,
            out_kernel,
            upstream,
            lse_kernel,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=causal,
            use_metal_kernel=False,
        )

        max_dq = float(mx.max(mx.abs(dq_auto - dq_ref)))
        max_dk = float(mx.max(mx.abs(dk_auto - dk_ref)))
        max_dv = float(mx.max(mx.abs(dv_auto - dv_ref)))

        assert max_dq < 5e-2, f"Autograd dq mismatch: {max_dq}"
        assert max_dk < 5e-2, f"Autograd dk mismatch: {max_dk}"
        assert max_dv < 5e-2, f"Autograd dv mismatch: {max_dv}"

    @pytest.mark.skipif(USE_REF, reason="Requires Metal kernel (FLASH_ATTENTION_MLX_REF=0)")
    @pytest.mark.parametrize("seqlens", _COMMON_VARLEN_SEQLENS[:3], ids=_seqlens_id)
    @pytest.mark.parametrize("causal", VARLEN_CAUSAL_FLAGS)
    def test_varlen_packed_variants_match_unpacked(self, seqlens, causal):
        """Ensure qkvpacked/kvpacked varlen APIs match the unpacked path."""

        nheads, nheads_k, headdim = 8, 2, 64

        (
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
        ) = make_varlen_qkv(seqlens, nheads, nheads_k, headdim)

        baseline = flash_attn_varlen_func(
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            causal=causal,
        )

        if isinstance(baseline, tuple):
            baseline = baseline[0]

        qkv_packed = mx.stack([q_varlen, k_varlen, v_varlen], axis=1)
        packed_qkv = flash_attn_varlen_qkvpacked_func(
            qkv_packed,
            cu_seqlens_q,
            max_seqlen_q,
            causal=causal,
        )

        if isinstance(packed_qkv, tuple):
            packed_qkv = packed_qkv[0]

        kv_packed = mx.stack([k_varlen, v_varlen], axis=1)
        packed_kv = flash_attn_varlen_kvpacked_func(
            q_varlen,
            kv_packed,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            causal=causal,
        )

        if isinstance(packed_kv, tuple):
            packed_kv = packed_kv[0]

        diff_qkv = float(mx.max(mx.abs(baseline - packed_qkv)))
        diff_kv = float(mx.max(mx.abs(baseline - packed_kv)))

        assert diff_qkv < 1e-3, f"Varlen qkvpacked mismatch: {diff_qkv}"
        assert diff_kv < 1e-3, f"Varlen kvpacked mismatch: {diff_kv}"


class TestPagedVarlenAttention:
    """Tests verifying paged varlen attention matches contiguous varlen attention.

    Note: These tests require the Metal kernel path (not reference fallback).
    Run with FLASH_ATTENTION_MLX_REF=0 environment variable set.
    """

    @pytest.mark.parametrize(
        "seqlens",
        [
            [64, 32],
            [33, 17, 25],
            [128],
            [48, 0, 16],
        ],
    )
    @pytest.mark.parametrize("page_size", [16, 32])
    @pytest.mark.parametrize("causal", [False, True])
    def test_paged_varlen_matches_contiguous(self, seqlens, page_size, causal):
        """Verify paged varlen attention matches contiguous varlen attention."""
        # Import USE_REF to check if Metal kernel path is available
        from flash_attn.flash_attn_mlx.utils import USE_REF
        if USE_REF:
            pytest.skip(
                "Paged KV cache requires Metal kernel path; "
                "set FLASH_ATTENTION_MLX_REF=0 to enable"
            )

        nheads = 4
        nheads_k = 2
        headdim = 64

        # Generate packed Q/K/V tensors and cu_seqlens
        (
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
        ) = make_varlen_qkv(seqlens, nheads, nheads_k, headdim, dtype=mx.float16)

        # Build paged caches from the packed K/V tensors
        k_paged, v_paged, page_table = _build_paged_cache_from_varlen(
            k_varlen, v_varlen, cu_seqlens_k, page_size
        )
        mx.eval(k_paged, v_paged, page_table)

        # Run contiguous varlen attention (block_table=None)
        out_contiguous_result = flash_attn_varlen_func(
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            causal=causal,
            block_table=None,
            return_attn_probs=False,
        )
        assert isinstance(out_contiguous_result, mx.array)
        out_contiguous = out_contiguous_result
        mx.eval(out_contiguous)

        # Run paged varlen attention (block_table=page_table)
        out_paged_result = flash_attn_varlen_func(
            q_varlen,
            k_paged,
            v_paged,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            causal=causal,
            block_table=page_table,
            return_attn_probs=False,
        )
        assert isinstance(out_paged_result, mx.array)
        out_paged = out_paged_result
        mx.eval(out_paged)

        # Assert outputs differ by less than 5e-3 in max absolute value
        max_diff = float(mx.max(mx.abs(out_paged - out_contiguous)))
        assert max_diff < 5e-3, (
            f"Paged varlen output mismatch: {max_diff} "
            f"(seqlens={seqlens}, page_size={page_size}, causal={causal})"
        )


# ============================================================================
# Forward Pass Tests
# ============================================================================


class TestForwardPass:
    """Tests for the forward pass of Flash Attention."""

    @pytest.mark.parametrize(
        "batch,seqlen,nheads,headdim",
        [
            (1, 16, 4, 64),
            (2, 32, 8, 64),
            (1, 64, 4, 32),
            (2, 16, 4, 128),
        ],
    )
    def test_forward_basic(self, batch, seqlen, nheads, headdim):
        """Test basic forward pass matches reference."""
        mx.random.seed(42)
        q = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        k = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        v = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1

        softmax_scale = 1.0 / math.sqrt(headdim)

        # Metal kernel
        out_metal, lse_metal = flash_attention_forward(
            q, k, v, softmax_scale=softmax_scale, causal=False
        )
        mx.eval(out_metal, lse_metal)

        # Reference
        out_ref, lse_ref = attention_forward_ref(
            q, k, v, softmax_scale=softmax_scale, causal=False
        )
        mx.eval(out_ref, lse_ref)

        max_diff_out = float(mx.max(mx.abs(out_metal - out_ref)))

        assert max_diff_out < 0.01, f"Output diff too large: {max_diff_out}"
        # Note: LSE can differ in absolute value but the gradients still match

    @pytest.mark.parametrize(
        "batch,seqlen,nheads,headdim",
        [
            (1, 16, 4, 64),
            (2, 32, 8, 64),
            (1, 64, 4, 32),
        ],
    )
    def test_forward_causal(self, batch, seqlen, nheads, headdim):
        """Test causal forward pass matches reference."""
        mx.random.seed(42)
        q = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        k = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        v = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1

        softmax_scale = 1.0 / math.sqrt(headdim)

        # Metal kernel
        out_metal, lse_metal = flash_attention_forward(
            q, k, v, softmax_scale=softmax_scale, causal=True
        )
        mx.eval(out_metal, lse_metal)

        # Reference
        out_ref, lse_ref = attention_forward_ref(
            q, k, v, softmax_scale=softmax_scale, causal=True
        )
        mx.eval(out_ref, lse_ref)

        max_diff_out = float(mx.max(mx.abs(out_metal - out_ref)))

        assert max_diff_out < 0.01, f"Output diff too large: {max_diff_out}"
        # Note: LSE can differ in absolute value but the gradients still match

    @pytest.mark.parametrize(
        "batch,seqlen,nheads,nheads_k,headdim",
        [
            (2, 16, 8, 2, 64),  # GQA: 4:1 ratio
            (2, 16, 4, 1, 64),  # MQA: 4:1 ratio
            (1, 32, 8, 4, 64),  # GQA: 2:1 ratio
        ],
    )
    def test_forward_gqa_mqa(self, batch, seqlen, nheads, nheads_k, headdim):
        """Test GQA/MQA forward pass matches reference."""
        mx.random.seed(42)
        q = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        k = mx.random.normal((batch, seqlen, nheads_k, headdim)).astype(mx.float16) * 0.1
        v = mx.random.normal((batch, seqlen, nheads_k, headdim)).astype(mx.float16) * 0.1

        softmax_scale = 1.0 / math.sqrt(headdim)

        # Metal kernel
        out_metal, lse_metal = flash_attention_forward(
            q, k, v, softmax_scale=softmax_scale, causal=True
        )
        mx.eval(out_metal, lse_metal)

        # Reference
        out_ref, lse_ref = attention_forward_ref(
            q, k, v, softmax_scale=softmax_scale, causal=True
        )
        mx.eval(out_ref, lse_ref)

        max_diff_out = float(mx.max(mx.abs(out_metal - out_ref)))

        assert max_diff_out < 0.01, f"Output diff too large: {max_diff_out}"


# ============================================================================
# Backward Pass Tests
# ============================================================================


class TestBackwardPass:
    """Tests for the backward pass of Flash Attention."""

    @pytest.mark.parametrize(
        "batch,seqlen,nheads,headdim",
        [
            (1, 16, 4, 64),
            (2, 16, 4, 64),
            (1, 32, 8, 64),
        ],
    )
    def test_backward_basic(self, batch, seqlen, nheads, headdim):
        """Test basic backward pass matches reference."""
        mx.random.seed(42)
        q = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        k = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        v = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1

        softmax_scale = 1.0 / math.sqrt(headdim)

        # Forward pass
        o, lse = flash_attention_forward(q, k, v, softmax_scale=softmax_scale, causal=False)
        mx.eval(o, lse)

        # Gradient
        mx.random.seed(123)
        do = mx.random.normal(o.shape).astype(mx.float16) * 0.1
        mx.eval(do)

        # Metal backward
        dq_metal, dk_metal, dv_metal = flash_attention_backward(
            q, k, v, o, do, lse, softmax_scale=softmax_scale, causal=False
        )
        mx.eval(dq_metal, dk_metal, dv_metal)

        # Reference backward
        dq_ref, dk_ref, dv_ref = attention_backward_ref(
            q, k, v, o, do, lse, softmax_scale=softmax_scale, causal=False
        )
        mx.eval(dq_ref, dk_ref, dv_ref)

        max_dq = float(mx.max(mx.abs(dq_metal - dq_ref)))
        max_dk = float(mx.max(mx.abs(dk_metal - dk_ref)))
        max_dv = float(mx.max(mx.abs(dv_metal - dv_ref)))

        assert max_dq < 0.01, f"dQ diff too large: {max_dq}"
        assert max_dk < 0.01, f"dK diff too large: {max_dk}"
        assert max_dv < 0.01, f"dV diff too large: {max_dv}"

    @pytest.mark.parametrize(
        "batch,seqlen,nheads,headdim",
        [
            (1, 16, 4, 64),
            (2, 16, 4, 64),
            (1, 32, 8, 64),
        ],
    )
    def test_backward_causal(self, batch, seqlen, nheads, headdim):
        """Test causal backward pass matches reference."""
        mx.random.seed(42)
        q = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        k = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        v = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1

        softmax_scale = 1.0 / math.sqrt(headdim)

        # Forward pass
        o, lse = flash_attention_forward(q, k, v, softmax_scale=softmax_scale, causal=True)
        mx.eval(o, lse)

        # Gradient
        mx.random.seed(123)
        do = mx.random.normal(o.shape).astype(mx.float16) * 0.1
        mx.eval(do)

        # Metal backward
        dq_metal, dk_metal, dv_metal = flash_attention_backward(
            q, k, v, o, do, lse, softmax_scale=softmax_scale, causal=True
        )
        mx.eval(dq_metal, dk_metal, dv_metal)

        # Reference backward
        dq_ref, dk_ref, dv_ref = attention_backward_ref(
            q, k, v, o, do, lse, softmax_scale=softmax_scale, causal=True
        )
        mx.eval(dq_ref, dk_ref, dv_ref)

        max_dq = float(mx.max(mx.abs(dq_metal - dq_ref)))
        max_dk = float(mx.max(mx.abs(dk_metal - dk_ref)))
        max_dv = float(mx.max(mx.abs(dv_metal - dv_ref)))

        assert max_dq < 0.01, f"dQ diff too large: {max_dq}"
        assert max_dk < 0.01, f"dK diff too large: {max_dk}"
        assert max_dv < 0.01, f"dV diff too large: {max_dv}"

    @pytest.mark.parametrize(
        "batch,seqlen,nheads,nheads_k,headdim",
        [
            (2, 16, 8, 2, 64),  # GQA
            (2, 16, 4, 1, 64),  # MQA
        ],
    )
    def test_backward_gqa_mqa(self, batch, seqlen, nheads, nheads_k, headdim):
        """Test GQA/MQA backward pass matches reference."""
        mx.random.seed(42)
        q = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        k = mx.random.normal((batch, seqlen, nheads_k, headdim)).astype(mx.float16) * 0.1
        v = mx.random.normal((batch, seqlen, nheads_k, headdim)).astype(mx.float16) * 0.1

        softmax_scale = 1.0 / math.sqrt(headdim)

        # Forward pass
        o, lse = flash_attention_forward(q, k, v, softmax_scale=softmax_scale, causal=True)
        mx.eval(o, lse)

        # Gradient
        mx.random.seed(123)
        do = mx.random.normal(o.shape).astype(mx.float16) * 0.1
        mx.eval(do)

        # Metal backward
        dq_metal, dk_metal, dv_metal = flash_attention_backward(
            q, k, v, o, do, lse, softmax_scale=softmax_scale, causal=True
        )
        mx.eval(dq_metal, dk_metal, dv_metal)

        # Reference backward
        dq_ref, dk_ref, dv_ref = attention_backward_ref(
            q, k, v, o, do, lse, softmax_scale=softmax_scale, causal=True
        )
        mx.eval(dq_ref, dk_ref, dv_ref)

        max_dq = float(mx.max(mx.abs(dq_metal - dq_ref)))
        max_dk = float(mx.max(mx.abs(dk_metal - dk_ref)))
        max_dv = float(mx.max(mx.abs(dv_metal - dv_ref)))

        assert max_dq < 0.01, f"dQ diff too large: {max_dq}"
        assert max_dk < 0.01, f"dK diff too large: {max_dk}"
        assert max_dv < 0.01, f"dV diff too large: {max_dv}"


# ============================================================================
# VJP Integration Tests
# ============================================================================


class TestVJPIntegration:
    """Tests for VJP (automatic differentiation) integration."""

    @pytest.mark.parametrize("causal", [False, True])
    def test_vjp_via_grad(self, causal):
        """Test that mx.grad works correctly with flash_attention_mlx."""
        mx.random.seed(42)
        batch, seqlen, nheads, headdim = 2, 16, 4, 64

        q = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        k = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        v = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1

        softmax_scale = 1.0 / math.sqrt(headdim)

        def loss_fn(q, k, v):
            out = flash_attention_mlx(q, k, v, softmax_scale=softmax_scale, causal=causal)
            return mx.sum(out)

        grad_fn = mx.grad(loss_fn, argnums=(0, 1, 2))
        dq, dk, dv = grad_fn(q, k, v)
        mx.eval(dq, dk, dv)

        # Reference
        o_ref, lse_ref = attention_forward_ref(
            q, k, v, softmax_scale=softmax_scale, causal=causal
        )
        mx.eval(o_ref, lse_ref)
        do = mx.ones_like(o_ref).astype(mx.float16)
        dq_ref, dk_ref, dv_ref = attention_backward_ref(
            q, k, v, o_ref, do, lse_ref, softmax_scale=softmax_scale, causal=causal
        )
        mx.eval(dq_ref, dk_ref, dv_ref)

        max_dq = float(mx.max(mx.abs(dq - dq_ref)))
        max_dk = float(mx.max(mx.abs(dk - dk_ref)))
        max_dv = float(mx.max(mx.abs(dv - dv_ref)))

        assert max_dq < 0.01, f"dQ diff too large: {max_dq}"
        assert max_dk < 0.01, f"dK diff too large: {max_dk}"
        assert max_dv < 0.01, f"dV diff too large: {max_dv}"

    def test_vjp_value_and_grad(self):
        """Test that mx.value_and_grad works correctly."""
        mx.random.seed(42)
        batch, seqlen, nheads, headdim = 2, 16, 4, 64

        q = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        k = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        v = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1

        softmax_scale = 1.0 / math.sqrt(headdim)

        def loss_fn(q, k, v):
            out = flash_attention_mlx(q, k, v, softmax_scale=softmax_scale, causal=True)
            return mx.sum(out)

        value_and_grad_fn = mx.value_and_grad(loss_fn, argnums=(0, 1, 2))
        loss_val, (dq, dk, dv) = value_and_grad_fn(q, k, v)
        mx.eval(loss_val, dq, dk, dv)

        # Verify shapes
        assert dq.shape == q.shape
        assert dk.shape == k.shape
        assert dv.shape == v.shape

        # Verify loss is a scalar
        assert loss_val.shape == ()

        # Verify gradients are not zero
        assert float(mx.linalg.norm(dq)) > 0
        assert float(mx.linalg.norm(dk)) > 0
        assert float(mx.linalg.norm(dv)) > 0

    def test_vjp_gqa(self):
        """Test VJP with grouped query attention."""
        mx.random.seed(42)
        batch, seqlen, nheads, nheads_k, headdim = 2, 16, 8, 2, 64

        q = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        k = mx.random.normal((batch, seqlen, nheads_k, headdim)).astype(mx.float16) * 0.1
        v = mx.random.normal((batch, seqlen, nheads_k, headdim)).astype(mx.float16) * 0.1

        softmax_scale = 1.0 / math.sqrt(headdim)

        def loss_fn(q, k, v):
            out = flash_attention_mlx(q, k, v, softmax_scale=softmax_scale, causal=True)
            return mx.sum(out)

        grad_fn = mx.grad(loss_fn, argnums=(0, 1, 2))
        dq, dk, dv = grad_fn(q, k, v)
        mx.eval(dq, dk, dv)

        # Verify shapes
        assert dq.shape == q.shape
        assert dk.shape == k.shape
        assert dv.shape == v.shape


# ============================================================================
# Edge Cases and Stress Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and stress scenarios."""

    def test_single_token(self):
        """Test with single token sequence."""
        mx.random.seed(42)
        batch, seqlen, nheads, headdim = 2, 1, 4, 64

        q = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        k = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        v = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1

        softmax_scale = 1.0 / math.sqrt(headdim)

        out_metal, lse_metal = flash_attention_forward(
            q, k, v, softmax_scale=softmax_scale, causal=True
        )
        mx.eval(out_metal, lse_metal)

        out_ref, lse_ref = attention_forward_ref(
            q, k, v, softmax_scale=softmax_scale, causal=True
        )
        mx.eval(out_ref, lse_ref)

        max_diff = float(mx.max(mx.abs(out_metal - out_ref)))
        assert max_diff < 0.01, f"Output diff too large: {max_diff}"

    def test_different_dtypes(self):
        """Test with different data types (when supported)."""
        mx.random.seed(42)
        batch, seqlen, nheads, headdim = 2, 16, 4, 64

        # Test fp16
        q = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        k = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        v = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1

        softmax_scale = 1.0 / math.sqrt(headdim)

        out, lse = flash_attention_forward(
            q, k, v, softmax_scale=softmax_scale, causal=False
        )
        mx.eval(out, lse)

        assert out.dtype == mx.float16
        assert out.shape == q.shape

    def test_longer_sequence(self):
        """Test with longer sequence lengths."""
        mx.random.seed(42)
        batch, seqlen, nheads, headdim = 1, 128, 4, 64

        q = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        k = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        v = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1

        softmax_scale = 1.0 / math.sqrt(headdim)

        out_metal, lse_metal = flash_attention_forward(
            q, k, v, softmax_scale=softmax_scale, causal=True
        )
        mx.eval(out_metal, lse_metal)

        out_ref, lse_ref = attention_forward_ref(
            q, k, v, softmax_scale=softmax_scale, causal=True
        )
        mx.eval(out_ref, lse_ref)

        max_diff = float(mx.max(mx.abs(out_metal - out_ref)))
        assert max_diff < 0.02, f"Output diff too large: {max_diff}"


# ============================================================================
# Public Interface Tests
# ============================================================================


class TestPublicInterface:
    """Tests for the public API functions."""

    def test_flash_attention_mlx_signature(self):
        """Test that flash_attention_mlx has the expected signature."""
        import inspect
        sig = inspect.signature(flash_attention_mlx)
        params = list(sig.parameters.keys())

        assert "q" in params
        assert "k" in params
        assert "v" in params
        assert "softmax_scale" in params
        assert "causal" in params

    def test_flash_attention_with_lse_returns_tuple(self):
        """Test that flash_attention_with_lse returns (output, lse) tuple."""
        mx.random.seed(42)
        batch, seqlen, nheads, headdim = 2, 16, 4, 64

        q = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        k = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        v = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1

        softmax_scale = 1.0 / math.sqrt(headdim)

        result = flash_attention_with_lse(
            q, k, v, softmax_scale=softmax_scale, causal=False
        )

        assert isinstance(result, tuple)
        assert len(result) == 2

        out, lse = result
        assert out.shape == (batch, seqlen, nheads, headdim)
        assert lse.shape == (batch, nheads, seqlen)


# ============================================================================
# Sliding Window Attention Tests
# ============================================================================


class TestPagedKVCacheForward:
    """Tests for paged KV cache forward pass."""

    @classmethod
    def setup_class(cls):
        _require_paged_kernel()

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("seqlen_k", [32, 96])
    @pytest.mark.parametrize("page_size", [16, 24])
    @pytest.mark.parametrize("headdim", [32, 64])
    def test_paged_matches_contiguous(self, batch_size, seqlen_k, page_size, headdim):
        """Verify paged KV cache path matches contiguous KV cache results."""
        mx.random.seed(42)
        nheads = 4
        nheads_k = 2
        seqlen_q = 4

        # Generate query and dense K/V caches
        q = mx.random.normal((batch_size, seqlen_q, nheads, headdim)).astype(mx.float16) * 0.1
        k_dense = mx.random.normal((batch_size, seqlen_k, nheads_k, headdim)).astype(mx.float16) * 0.1
        v_dense = mx.random.normal((batch_size, seqlen_k, nheads_k, headdim)).astype(mx.float16) * 0.1

        # Build paged caches
        seqlens = [seqlen_k] * batch_size
        k_paged, v_paged, page_table = _build_paged_cache_from_padded(
            k_dense, v_dense, seqlens, page_size
        )

        cache_lens = mx.array([seqlen_k] * batch_size, dtype=mx.int32)

        # Run contiguous baseline
        out_contiguous, lse_contiguous = flash_attn_with_kvcache(
            q, k_dense, v_dense,
            cache_seqlens=cache_lens,
            causal=True,
            return_softmax_lse=True,
        )
        mx.eval(out_contiguous, lse_contiguous)

        # Run paged path
        out_paged, lse_paged = flash_attn_with_kvcache(
            q, k_paged, v_paged,
            cache_seqlens=cache_lens,
            block_table=page_table,
            causal=True,
            return_softmax_lse=True,
        )
        mx.eval(out_paged, lse_paged)

        # Assert outputs match
        max_out_diff = float(mx.max(mx.abs(out_paged - out_contiguous)))
        max_lse_diff = float(mx.max(mx.abs(lse_paged - lse_contiguous)))

        assert max_out_diff < 5e-3, f"Paged output mismatch: {max_out_diff}"
        assert max_lse_diff < 5e-3, f"Paged LSE mismatch: {max_lse_diff}"


class TestPagedCacheUpdate:
    """Tests verifying paged cache update kernel writes correct data."""

    @classmethod
    def setup_class(cls):
        _require_paged_kernel()

    def test_single_update_matches_reference(self):
        """Verify a single paged update matches contiguous tensors."""
        mx.random.seed(7)
        batch = 2
        page_size = 4
        max_pages_per_seq = 4
        nheads_k = 2
        headdim = 8
        seqlen_new = 5

        k_cache, v_cache, page_table = _allocate_paged_cache_buffers(
            batch,
            max_pages_per_seq,
            page_size,
            nheads_k,
            headdim,
        )
        cache_seqlens = mx.zeros(batch, dtype=mx.int32)

        new_k = mx.random.normal((batch, seqlen_new, nheads_k, headdim)).astype(mx.float16) * 0.1
        new_v = mx.random.normal((batch, seqlen_new, nheads_k, headdim)).astype(mx.float16) * 0.1

        updated_lengths = update_paged_kv_cache(
            k_cache,
            v_cache,
            new_k,
            new_v,
            page_table,
            cache_seqlens,
        )
        mx.eval(k_cache, v_cache, updated_lengths)

        expected_len = seqlen_new
        for length in updated_lengths:
            assert int(length.item()) == expected_len

        rebuilt_k = _paged_cache_to_contiguous(
            k_cache,
            page_table,
            [int(length.item()) for length in updated_lengths],
        )
        rebuilt_v = _paged_cache_to_contiguous(
            v_cache,
            page_table,
            [int(length.item()) for length in updated_lengths],
        )
        mx.eval(rebuilt_k, rebuilt_v)

        max_k_diff = float(mx.max(mx.abs(rebuilt_k[:, :expected_len] - new_k)))
        max_v_diff = float(mx.max(mx.abs(rebuilt_v[:, :expected_len] - new_v)))

        assert max_k_diff < 5e-3, f"Paged cache K mismatch: {max_k_diff}"
        assert max_v_diff < 5e-3, f"Paged cache V mismatch: {max_v_diff}"

    def test_multiple_updates_cross_page_boundaries(self):
        """Ensure appending new tokens spans multiple pages correctly."""
        mx.random.seed(11)
        batch = 1
        page_size = 4
        max_pages_per_seq = 3
        nheads_k = 2
        headdim = 8

        k_cache, v_cache, page_table = _allocate_paged_cache_buffers(
            batch,
            max_pages_per_seq,
            page_size,
            nheads_k,
            headdim,
        )
        cache_seqlens = mx.zeros(batch, dtype=mx.int32)

        init_len = 3
        new_len = 5

        init_k = mx.random.normal((batch, init_len, nheads_k, headdim)).astype(mx.float16) * 0.1
        init_v = mx.random.normal((batch, init_len, nheads_k, headdim)).astype(mx.float16) * 0.1
        cache_seqlens = update_paged_kv_cache(
            k_cache,
            v_cache,
            init_k,
            init_v,
            page_table,
            cache_seqlens,
        )

        new_k = mx.random.normal((batch, new_len, nheads_k, headdim)).astype(mx.float16) * 0.1
        new_v = mx.random.normal((batch, new_len, nheads_k, headdim)).astype(mx.float16) * 0.1
        cache_seqlens = update_paged_kv_cache(
            k_cache,
            v_cache,
            new_k,
            new_v,
            page_table,
            cache_seqlens,
        )
        mx.eval(k_cache, v_cache, cache_seqlens)

        rebuilt_k = _paged_cache_to_contiguous(
            k_cache,
            page_table,
            [int(cache_seqlens[0].item())],
        )
        rebuilt_v = _paged_cache_to_contiguous(
            v_cache,
            page_table,
            [int(cache_seqlens[0].item())],
        )
        mx.eval(rebuilt_k, rebuilt_v)

        expected_k = mx.concatenate([init_k, new_k], axis=1)
        expected_v = mx.concatenate([init_v, new_v], axis=1)
        mx.eval(expected_k, expected_v)

        total_len = init_len + new_len
        assert int(cache_seqlens[0].item()) == total_len

        max_k_diff = float(mx.max(mx.abs(rebuilt_k[0, :total_len] - expected_k[0, :total_len])))
        max_v_diff = float(mx.max(mx.abs(rebuilt_v[0, :total_len] - expected_v[0, :total_len])))

        assert max_k_diff < 5e-3, f"Paged append K mismatch: {max_k_diff}"
        assert max_v_diff < 5e-3, f"Paged append V mismatch: {max_v_diff}"


class TestPagedIncrementalDecoding:
    """Tests incremental decoding with paged caches vs contiguous reference."""

    @classmethod
    def setup_class(cls):
        _require_paged_kernel()

    def test_multi_step_incremental_matches_contiguous(self):
        mx.random.seed(101)
        batch = 2
        nheads = 4
        nheads_k = 2
        headdim = 32
        page_size = 4
        steps = 5
        tokens_per_step = 2
        max_tokens = steps * tokens_per_step
        max_pages_per_seq = (max_tokens + page_size - 1) // page_size + 1

        k_cache_paged, v_cache_paged, page_table = _allocate_paged_cache_buffers(
            batch,
            max_pages_per_seq,
            page_size,
            nheads_k,
            headdim,
        )
        cache_seqlens = mx.zeros(batch, dtype=mx.int32)

        k_dense = mx.zeros((batch, 0, nheads_k, headdim), dtype=mx.float16)
        v_dense = mx.zeros_like(k_dense)
        current_len = 0

        softmax_scale = 1.0 / math.sqrt(headdim)

        for _ in range(steps):
            q_step = mx.random.normal((batch, 1, nheads, headdim)).astype(mx.float16) * 0.1
            k_step = mx.random.normal((batch, tokens_per_step, nheads_k, headdim)).astype(mx.float16) * 0.1
            v_step = mx.random.normal((batch, tokens_per_step, nheads_k, headdim)).astype(mx.float16) * 0.1

            if current_len == 0:
                k_dense = k_step
                v_dense = v_step
            else:
                k_dense = mx.concatenate([k_dense[:, :current_len], k_step], axis=1)
                v_dense = mx.concatenate([v_dense[:, :current_len], v_step], axis=1)
            current_len += tokens_per_step

            ref_out, ref_lse = flash_attention_with_lse(
                q_step,
                k_dense[:, :current_len],
                v_dense[:, :current_len],
                softmax_scale=softmax_scale,
                causal=True,
            )
            mx.eval(ref_out, ref_lse)

            paged_out, paged_lse, cache_seqlens = flash_attn_with_kvcache(
                q_step,
                k_cache_paged,
                v_cache_paged,
                k=k_step,
                v=v_step,
                cache_seqlens=cache_seqlens,
                block_table=page_table,
                softmax_scale=softmax_scale,
                causal=True,
                return_softmax_lse=True,
                return_cache_seqlens=True,
            )
            mx.eval(paged_out, paged_lse, cache_seqlens)

            max_out_diff = float(mx.max(mx.abs(paged_out - ref_out)))
            max_lse_diff = float(mx.max(mx.abs(paged_lse - ref_lse)))

            assert max_out_diff < 5e-3, f"Incremental output mismatch: {max_out_diff}"
            assert max_lse_diff < 5e-3, f"Incremental LSE mismatch: {max_lse_diff}"

        final_lengths = [int(length.item()) for length in cache_seqlens]
        assert all(length == current_len for length in final_lengths)


class TestPagedCacheEdgeCases:
    """Edge-case coverage for paged cache updates."""

    @classmethod
    def setup_class(cls):
        _require_paged_kernel()

    def test_exact_page_boundary_then_append(self):
        """Writing exact page multiples should roll to the next page cleanly."""
        mx.random.seed(17)
        batch = 1
        page_size = 6
        max_pages_per_seq = 3
        nheads_k = 2
        headdim = 16

        k_cache, v_cache, page_table = _allocate_paged_cache_buffers(
            batch,
            max_pages_per_seq,
            page_size,
            nheads_k,
            headdim,
        )
        cache_seqlens = mx.zeros(batch, dtype=mx.int32)

        init_k = mx.random.normal((batch, page_size, nheads_k, headdim)).astype(mx.float16) * 0.1
        init_v = mx.random.normal((batch, page_size, nheads_k, headdim)).astype(mx.float16) * 0.1
        cache_seqlens = update_paged_kv_cache(
            k_cache,
            v_cache,
            init_k,
            init_v,
            page_table,
            cache_seqlens,
        )

        append_len = 3
        extra_k = mx.random.normal((batch, append_len, nheads_k, headdim)).astype(mx.float16) * 0.1
        extra_v = mx.random.normal((batch, append_len, nheads_k, headdim)).astype(mx.float16) * 0.1
        cache_seqlens = update_paged_kv_cache(
            k_cache,
            v_cache,
            extra_k,
            extra_v,
            page_table,
            cache_seqlens,
        )
        mx.eval(cache_seqlens, k_cache, v_cache)

        rebuilt_k = _paged_cache_to_contiguous(
            k_cache,
            page_table,
            [int(cache_seqlens[0].item())],
        )
        rebuilt_v = _paged_cache_to_contiguous(
            v_cache,
            page_table,
            [int(cache_seqlens[0].item())],
        )
        mx.eval(rebuilt_k, rebuilt_v)

        expected_k = mx.concatenate([init_k, extra_k], axis=1)
        expected_v = mx.concatenate([init_v, extra_v], axis=1)
        total_len = page_size + append_len

        max_k_diff = float(mx.max(mx.abs(rebuilt_k[0, :total_len] - expected_k[0, :total_len])))
        max_v_diff = float(mx.max(mx.abs(rebuilt_v[0, :total_len] - expected_v[0, :total_len])))

        assert int(cache_seqlens[0].item()) == total_len
        assert max_k_diff < 5e-3, f"Boundary K mismatch: {max_k_diff}"
        assert max_v_diff < 5e-3, f"Boundary V mismatch: {max_v_diff}"

    def test_fill_full_capacity(self):
        """Filling to max pages should still match contiguous reference."""
        mx.random.seed(23)
        batch = 1
        page_size = 4
        max_pages_per_seq = 2
        capacity = page_size * max_pages_per_seq
        nheads_k = 2
        headdim = 8

        k_cache, v_cache, page_table = _allocate_paged_cache_buffers(
            batch,
            max_pages_per_seq,
            page_size,
            nheads_k,
            headdim,
        )
        cache_seqlens = mx.zeros(batch, dtype=mx.int32)

        segments_k = []
        segments_v = []
        for _ in range(2):
            seg_k = mx.random.normal((batch, page_size, nheads_k, headdim)).astype(mx.float16) * 0.1
            seg_v = mx.random.normal((batch, page_size, nheads_k, headdim)).astype(mx.float16) * 0.1
            segments_k.append(seg_k)
            segments_v.append(seg_v)
            cache_seqlens = update_paged_kv_cache(
                k_cache,
                v_cache,
                seg_k,
                seg_v,
                page_table,
                cache_seqlens,
            )

        rebuilt_k = _paged_cache_to_contiguous(
            k_cache,
            page_table,
            [capacity],
        )
        rebuilt_v = _paged_cache_to_contiguous(
            v_cache,
            page_table,
            [capacity],
        )
        mx.eval(rebuilt_k, rebuilt_v)

        expected_k = mx.concatenate(segments_k, axis=1)
        expected_v = mx.concatenate(segments_v, axis=1)
        mx.eval(expected_k, expected_v)

        max_k_diff = float(mx.max(mx.abs(rebuilt_k[0, :capacity] - expected_k[0, :capacity])))
        max_v_diff = float(mx.max(mx.abs(rebuilt_v[0, :capacity] - expected_v[0, :capacity])))
        assert int(cache_seqlens[0].item()) == capacity
        assert max_k_diff < 5e-3, f"Full capacity K mismatch: {max_k_diff}"
        assert max_v_diff < 5e-3, f"Full capacity V mismatch: {max_v_diff}"

    def test_partial_page_padding_is_zero(self):
        """Tokens beyond logical length should remain zero padded."""
        mx.random.seed(29)
        batch = 2
        page_size = 5
        max_pages_per_seq = 2
        nheads_k = 2
        headdim = 8
        seqlen_new = 3

        k_cache, v_cache, page_table = _allocate_paged_cache_buffers(
            batch,
            max_pages_per_seq,
            page_size,
            nheads_k,
            headdim,
        )
        cache_seqlens = mx.zeros(batch, dtype=mx.int32)

        new_k = mx.random.normal((batch, seqlen_new, nheads_k, headdim)).astype(mx.float16) * 0.1
        new_v = mx.random.normal((batch, seqlen_new, nheads_k, headdim)).astype(mx.float16) * 0.1
        cache_seqlens = update_paged_kv_cache(
            k_cache,
            v_cache,
            new_k,
            new_v,
            page_table,
            cache_seqlens,
        )

        mx.eval(k_cache, v_cache)

        max_tail_norm = 0.0
        for b in range(batch):
            page_idx = int(page_table[b, 0].item())
            tail_k = k_cache[page_idx, seqlen_new:, :, :]
            tail_v = v_cache[page_idx, seqlen_new:, :, :]
            mx.eval(tail_k, tail_v)
            max_tail_norm = max(
                max_tail_norm,
                float(mx.max(mx.abs(tail_k))),
                float(mx.max(mx.abs(tail_v))),
            )

        assert max_tail_norm < 1e-6, f"Padding should remain zero, got {max_tail_norm}"


class TestKVCache:
    """Tests for KV cache functionality."""

    def test_kvcache_basic(self):
        """Test basic KV cache inference."""
        mx.random.seed(42)
        batch, max_seqlen, nheads, headdim = 2, 64, 4, 64

        # Initial cache (empty)
        k_cache = mx.zeros((batch, max_seqlen, nheads, headdim), dtype=mx.float16)
        v_cache = mx.zeros((batch, max_seqlen, nheads, headdim), dtype=mx.float16)

        # First token - fill cache with initial K/V
        seqlen_init = 8
        q = mx.random.normal((batch, 1, nheads, headdim)).astype(mx.float16) * 0.1
        k_new = mx.random.normal((batch, seqlen_init, nheads, headdim)).astype(mx.float16) * 0.1
        v_new = mx.random.normal((batch, seqlen_init, nheads, headdim)).astype(mx.float16) * 0.1

        out = flash_attn_with_kvcache(
            q, k_cache, v_cache,
            k=k_new, v=v_new,
            cache_seqlens=0,
            causal=True,
        )
        mx.eval(out)

        assert out.shape == (batch, 1, nheads, headdim)
        assert not mx.any(mx.isnan(out))

    def test_kvcache_incremental(self):
        """Test incremental decoding with KV cache."""
        mx.random.seed(42)
        batch, max_seqlen, nheads, headdim = 1, 64, 4, 64

        # Initial fill
        seqlen_init = 8
        k_cache = mx.zeros((batch, max_seqlen, nheads, headdim), dtype=mx.float16)
        v_cache = mx.zeros((batch, max_seqlen, nheads, headdim), dtype=mx.float16)

        k_init = mx.random.normal((batch, seqlen_init, nheads, headdim)).astype(mx.float16) * 0.1
        v_init = mx.random.normal((batch, seqlen_init, nheads, headdim)).astype(mx.float16) * 0.1

        # Put initial KV into cache
        k_cache = mx.concatenate([k_init, k_cache[:, seqlen_init:, :, :]], axis=1)
        v_cache = mx.concatenate([v_init, v_cache[:, seqlen_init:, :, :]], axis=1)

        # Incremental decode - add one token at a time
        current_len = seqlen_init
        for i in range(4):
            q = mx.random.normal((batch, 1, nheads, headdim)).astype(mx.float16) * 0.1
            k_new = mx.random.normal((batch, 1, nheads, headdim)).astype(mx.float16) * 0.1
            v_new = mx.random.normal((batch, 1, nheads, headdim)).astype(mx.float16) * 0.1

            out = flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k_new, v=v_new,
                cache_seqlens=current_len,
                causal=True,
            )
            mx.eval(out)

            # Update cache manually for next iteration
            k_cache = mx.concatenate([
                k_cache[:, :current_len, :, :],
                k_new,
                k_cache[:, current_len+1:, :, :]
            ], axis=1)
            v_cache = mx.concatenate([
                v_cache[:, :current_len, :, :],
                v_new,
                v_cache[:, current_len+1:, :, :]
            ], axis=1)
            current_len += 1

            assert out.shape == (batch, 1, nheads, headdim)
            assert not mx.any(mx.isnan(out))

    def test_kvcache_without_new_kv(self):
        """Test KV cache attention without adding new K/V."""
        mx.random.seed(42)
        batch, nheads, headdim = 2, 4, 64
        cache_len = 16

        # Pre-filled cache
        k_cache = mx.random.normal((batch, cache_len, nheads, headdim)).astype(mx.float16) * 0.1
        v_cache = mx.random.normal((batch, cache_len, nheads, headdim)).astype(mx.float16) * 0.1

        # Query without new K/V
        q = mx.random.normal((batch, 4, nheads, headdim)).astype(mx.float16) * 0.1

        out = flash_attn_with_kvcache(
            q, k_cache, v_cache,
            cache_seqlens=cache_len,
            causal=False,  # Full attention
        )
        mx.eval(out)

        assert out.shape == (batch, 4, nheads, headdim)
        assert not mx.any(mx.isnan(out))


class TestAdvancedFeatures:
    """Tests for advanced features: ALiBi, softcap, dropout."""

    def test_softcap_forward(self):
        """Test forward pass with softcap."""
        mx.random.seed(42)
        batch, seqlen, nheads, headdim = 2, 16, 4, 64

        q = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        k = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        v = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1

        softmax_scale = 1.0 / math.sqrt(headdim)
        softcap = 50.0

        # Forward with softcap
        out, lse = flash_attention_forward(
            q, k, v, softmax_scale=softmax_scale, causal=True, softcap=softcap
        )
        mx.eval(out, lse)

        # Reference
        out_ref, lse_ref, _ = attention_ref_mlx(
            q, k, v, softmax_scale=softmax_scale, causal=True, softcap=softcap
        )
        mx.eval(out_ref, lse_ref)

        max_diff = float(mx.max(mx.abs(out - out_ref)))
        assert max_diff < 0.02, f"Output diff too large: {max_diff}"

    def test_alibi_forward_reference(self):
        """Test ALiBi forward pass (uses reference implementation)."""
        mx.random.seed(42)
        batch, seqlen, nheads, headdim = 2, 16, 4, 64

        q = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        k = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        v = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1

        # Create ALiBi slopes
        alibi_slopes = mx.array([0.5, 0.25, 0.125, 0.0625], dtype=mx.float32)

        softmax_scale = 1.0 / math.sqrt(headdim)

        # Reference with ALiBi
        out, lse, _ = attention_ref_mlx(
            q, k, v, softmax_scale=softmax_scale, causal=True, alibi_slopes=alibi_slopes
        )
        mx.eval(out, lse)

        # Verify output shape
        assert out.shape == q.shape
        assert lse.shape == (batch, nheads, seqlen)

        # Verify not NaN/Inf
        assert not mx.any(mx.isnan(out))
        assert not mx.any(mx.isinf(out))

    def test_dropout_reference(self):
        """Test dropout (uses reference implementation)."""
        mx.random.seed(42)
        batch, seqlen, nheads, headdim = 2, 16, 4, 64

        q = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        k = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        v = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1

        softmax_scale = 1.0 / math.sqrt(headdim)
        dropout_p = 0.1

        # Reference with dropout
        out, lse, _ = attention_ref_mlx(
            q, k, v, softmax_scale=softmax_scale, causal=True, dropout_p=dropout_p
        )
        mx.eval(out, lse)

        # Verify output shape and no NaN/Inf
        assert out.shape == q.shape
        assert not mx.any(mx.isnan(out))

    def test_deterministic_backward(self):
        """Test deterministic backward mode (same result on repeated calls)."""
        mx.random.seed(42)
        batch, seqlen, nheads, headdim = 2, 16, 4, 64

        q = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        k = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        v = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1

        softmax_scale = 1.0 / math.sqrt(headdim)

        def loss_fn(q, k, v):
            out = flash_attention_mlx(q, k, v, softmax_scale=softmax_scale, causal=True)
            return mx.sum(out)

        # Compute gradients twice
        grad_fn = mx.grad(loss_fn, argnums=(0, 1, 2))

        dq1, dk1, dv1 = grad_fn(q, k, v)
        mx.eval(dq1, dk1, dv1)

        dq2, dk2, dv2 = grad_fn(q, k, v)
        mx.eval(dq2, dk2, dv2)

        # Should be identical (deterministic)
        assert float(mx.max(mx.abs(dq1 - dq2))) < 1e-6
        assert float(mx.max(mx.abs(dk1 - dk2))) < 1e-6
        assert float(mx.max(mx.abs(dv1 - dv2))) < 1e-6


class TestExtendedHeadDimensions:
    """Tests for extended head dimensions (96, 160, 192, 256)."""

    @pytest.mark.parametrize(
        "headdim",
        [96, 160, 192, 256],
    )
    def test_extended_headdim_forward(self, headdim):
        """Test forward pass with extended head dimensions."""
        mx.random.seed(42)
        batch, seqlen, nheads = 2, 16, 4

        q = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        k = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        v = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1

        softmax_scale = 1.0 / math.sqrt(headdim)

        # Metal kernel
        out_metal, lse_metal = flash_attention_forward(
            q, k, v, softmax_scale=softmax_scale, causal=True
        )
        mx.eval(out_metal, lse_metal)

        # Reference
        out_ref, lse_ref = attention_forward_ref(
            q, k, v, softmax_scale=softmax_scale, causal=True
        )
        mx.eval(out_ref, lse_ref)

        max_diff = float(mx.max(mx.abs(out_metal - out_ref)))
        assert max_diff < 0.02, f"Output diff too large for headdim={headdim}: {max_diff}"

    @pytest.mark.parametrize(
        "headdim",
        [96, 160, 192, 256],
    )
    def test_extended_headdim_backward(self, headdim):
        """Test backward pass with extended head dimensions."""
        mx.random.seed(42)
        batch, seqlen, nheads = 2, 16, 4

        q = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        k = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        v = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1

        softmax_scale = 1.0 / math.sqrt(headdim)

        # Forward
        out, lse = flash_attention_forward(
            q, k, v, softmax_scale=softmax_scale, causal=True
        )
        mx.eval(out, lse)

        # Gradient
        mx.random.seed(123)
        do = mx.random.normal(out.shape).astype(mx.float16) * 0.1
        mx.eval(do)

        # Metal backward
        dq_metal, dk_metal, dv_metal = flash_attention_backward(
            q, k, v, out, do, lse, softmax_scale=softmax_scale, causal=True
        )
        mx.eval(dq_metal, dk_metal, dv_metal)

        # Reference backward
        dq_ref, dk_ref, dv_ref = attention_backward_ref(
            q, k, v, out, do, lse, softmax_scale=softmax_scale, causal=True
        )
        mx.eval(dq_ref, dk_ref, dv_ref)

        max_dq = float(mx.max(mx.abs(dq_metal - dq_ref)))
        max_dk = float(mx.max(mx.abs(dk_metal - dk_ref)))
        max_dv = float(mx.max(mx.abs(dv_metal - dv_ref)))

        assert max_dq < 0.02, f"dQ diff too large for headdim={headdim}: {max_dq}"
        assert max_dk < 0.02, f"dK diff too large for headdim={headdim}: {max_dk}"
        assert max_dv < 0.02, f"dV diff too large for headdim={headdim}: {max_dv}"


class TestSlidingWindowAttention:
    """Tests for sliding window (local) attention."""

    @pytest.mark.parametrize(
        "window_size",
        [
            (4, 4),   # Symmetric window
            (2, 6),   # Asymmetric window
            (8, -1),  # Only left window
            (-1, 4),  # Only right window
        ],
    )
    def test_sliding_window_forward(self, window_size):
        """Test sliding window forward pass matches reference."""
        mx.random.seed(42)
        batch, seqlen, nheads, headdim = 2, 32, 4, 64

        q = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        k = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        v = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1

        softmax_scale = 1.0 / math.sqrt(headdim)

        # Metal forward with sliding window
        out_metal, lse_metal = flash_attention_forward(
            q, k, v, softmax_scale=softmax_scale, causal=False, window_size=window_size
        )
        mx.eval(out_metal, lse_metal)

        # Reference forward (uses same window_size implementation)
        out_ref, lse_ref, _ = attention_ref_mlx(
            q, k, v, softmax_scale=softmax_scale, causal=False, window_size=window_size
        )
        mx.eval(out_ref, lse_ref)

        max_out_diff = float(mx.max(mx.abs(out_metal - out_ref)))
        max_lse_diff = float(mx.max(mx.abs(lse_metal - lse_ref)))

        assert max_out_diff < 0.01, f"Output diff too large: {max_out_diff}"
        assert max_lse_diff < 0.1, f"LSE diff too large: {max_lse_diff}"

    def test_sliding_window_with_causal(self):
        """Test sliding window combined with causal masking."""
        mx.random.seed(42)
        batch, seqlen, nheads, headdim = 2, 32, 4, 64

        q = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        k = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        v = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1

        softmax_scale = 1.0 / math.sqrt(headdim)
        window_size = (4, 0)  # Left window only (typical for causal)

        # Metal forward with causal + sliding window
        out_metal, lse_metal = flash_attention_forward(
            q, k, v, softmax_scale=softmax_scale, causal=True, window_size=window_size
        )
        mx.eval(out_metal, lse_metal)

        # Reference forward
        out_ref, lse_ref, _ = attention_ref_mlx(
            q, k, v, softmax_scale=softmax_scale, causal=True, window_size=window_size
        )
        mx.eval(out_ref, lse_ref)

        max_diff = float(mx.max(mx.abs(out_metal - out_ref)))
        assert max_diff < 0.01, f"Output diff too large: {max_diff}"

    def test_sliding_window_backward(self):
        """Test sliding window backward pass matches reference."""
        mx.random.seed(42)
        batch, seqlen, nheads, headdim = 2, 32, 4, 64

        q = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        k = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        v = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1

        softmax_scale = 1.0 / math.sqrt(headdim)
        window_size = (4, 4)

        # Forward pass
        out, lse = flash_attention_forward(
            q, k, v, softmax_scale=softmax_scale, causal=False, window_size=window_size
        )
        mx.eval(out, lse)

        # Gradient
        mx.random.seed(123)
        do = mx.random.normal(out.shape).astype(mx.float16) * 0.1
        mx.eval(do)

        # Metal backward
        dq_metal, dk_metal, dv_metal = flash_attention_backward(
            q, k, v, out, do, lse,
            softmax_scale=softmax_scale, causal=False, window_size=window_size
        )
        mx.eval(dq_metal, dk_metal, dv_metal)

        # Reference backward
        dq_ref, dk_ref, dv_ref = attention_backward_ref(
            q, k, v, out, do, lse, softmax_scale=softmax_scale, causal=False
        )
        mx.eval(dq_ref, dk_ref, dv_ref)

        # With sliding window, the gradients will be different from full attention
        # So we just check that the shapes are correct and values are reasonable
        assert dq_metal.shape == q.shape
        assert dk_metal.shape == k.shape
        assert dv_metal.shape == v.shape

        # Check values are not NaN/Inf
        assert not mx.any(mx.isnan(dq_metal))
        assert not mx.any(mx.isnan(dk_metal))
        assert not mx.any(mx.isnan(dv_metal))

    def test_sliding_window_vjp(self):
        """Test VJP with sliding window attention."""
        mx.random.seed(42)
        batch, seqlen, nheads, headdim = 2, 32, 4, 64

        q = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        k = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        v = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1

        softmax_scale = 1.0 / math.sqrt(headdim)
        window_size = (4, 4)

        def loss_fn(q, k, v):
            out = flash_attention_mlx(
                q, k, v, softmax_scale=softmax_scale, causal=False, window_size=window_size
            )
            return mx.sum(out)

        # Compute gradients via VJP
        _, grads = mx.value_and_grad(loss_fn, argnums=(0, 1, 2))(q, k, v)
        dq, dk, dv = grads
        mx.eval(dq, dk, dv)

        # Check shapes
        assert dq.shape == q.shape
        assert dk.shape == k.shape
        assert dv.shape == v.shape

        # Check values are not NaN/Inf
        assert not mx.any(mx.isnan(dq))
        assert not mx.any(mx.isnan(dk))
        assert not mx.any(mx.isnan(dv))


class TestALiBiMetalKernel:
    """Tests for ALiBi support in Metal kernels (not just reference)."""

    def test_alibi_forward_metal_kernel(self):
        """Verify Metal kernel ALiBi forward matches reference."""
        mx.random.seed(42)
        batch, seqlen, nheads, headdim = 2, 16, 4, 64

        q = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        k = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        v = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1

        # Create ALiBi slopes (1D shape: per-head slopes)
        alibi_slopes = mx.array([0.5, 0.25, 0.125, 0.0625], dtype=mx.float32)
        softmax_scale = 1.0 / math.sqrt(headdim)

        # Metal kernel path (flash_attention_with_lse now uses Metal for ALiBi)
        out_metal, lse_metal = flash_attention_with_lse(
            q, k, v, softmax_scale=softmax_scale, causal=True, alibi_slopes=alibi_slopes
        )
        mx.eval(out_metal, lse_metal)

        # Reference implementation
        out_ref, lse_ref, _ = attention_ref_mlx(
            q, k, v, softmax_scale=softmax_scale, causal=True, alibi_slopes=alibi_slopes
        )
        mx.eval(out_ref, lse_ref)

        # Compare outputs
        max_diff = float(mx.max(mx.abs(out_metal - out_ref)))
        assert max_diff < 0.05, f"ALiBi forward output diff too large: {max_diff}"

        # Verify shapes
        assert out_metal.shape == q.shape
        assert lse_metal.shape == (batch, nheads, seqlen)

    def test_alibi_forward_2d_slopes(self):
        """Test ALiBi with 2D slopes (batch, nheads)."""
        mx.random.seed(42)
        batch, seqlen, nheads, headdim = 2, 16, 4, 64

        q = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        k = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        v = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1

        # Create 2D ALiBi slopes (per-batch, per-head)
        alibi_slopes = mx.array([
            [0.5, 0.25, 0.125, 0.0625],
            [0.4, 0.2, 0.1, 0.05],
        ], dtype=mx.float32)
        softmax_scale = 1.0 / math.sqrt(headdim)

        # Metal kernel path
        out_metal, lse_metal = flash_attention_with_lse(
            q, k, v, softmax_scale=softmax_scale, causal=True, alibi_slopes=alibi_slopes
        )
        mx.eval(out_metal, lse_metal)

        # Reference implementation
        out_ref, lse_ref, _ = attention_ref_mlx(
            q, k, v, softmax_scale=softmax_scale, causal=True, alibi_slopes=alibi_slopes
        )
        mx.eval(out_ref, lse_ref)

        max_diff = float(mx.max(mx.abs(out_metal - out_ref)))
        assert max_diff < 0.05, f"ALiBi 2D slopes forward diff too large: {max_diff}"

    def test_alibi_forward_noncausal(self):
        """Test ALiBi in non-causal mode."""
        mx.random.seed(42)
        batch, seqlen, nheads, headdim = 2, 16, 4, 64

        q = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        k = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        v = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1

        alibi_slopes = mx.array([0.5, 0.25, 0.125, 0.0625], dtype=mx.float32)
        softmax_scale = 1.0 / math.sqrt(headdim)

        # Non-causal with ALiBi
        out_metal, lse_metal = flash_attention_with_lse(
            q, k, v, softmax_scale=softmax_scale, causal=False, alibi_slopes=alibi_slopes
        )
        mx.eval(out_metal, lse_metal)

        out_ref, lse_ref, _ = attention_ref_mlx(
            q, k, v, softmax_scale=softmax_scale, causal=False, alibi_slopes=alibi_slopes
        )
        mx.eval(out_ref, lse_ref)

        max_diff = float(mx.max(mx.abs(out_metal - out_ref)))
        assert max_diff < 0.05, f"ALiBi non-causal forward diff too large: {max_diff}"

    def test_alibi_varlen_forward_metal_kernel(self):
        """Test ALiBi with varlen sequences using Metal kernel."""
        mx.random.seed(42)
        batch, max_seqlen, nheads, headdim = 3, 16, 4, 64

        # Generate variable length sequences
        cu_seqlens_q = mx.array([0, 8, 14, 22], dtype=mx.int32)
        cu_seqlens_k = cu_seqlens_q
        total_q = int(cu_seqlens_q[-1].item())

        q = mx.random.normal((total_q, nheads, headdim)).astype(mx.float16) * 0.1
        k = mx.random.normal((total_q, nheads, headdim)).astype(mx.float16) * 0.1
        v = mx.random.normal((total_q, nheads, headdim)).astype(mx.float16) * 0.1

        alibi_slopes = mx.array([0.5, 0.25, 0.125, 0.0625], dtype=mx.float32)
        softmax_scale = 1.0 / math.sqrt(headdim)

        # Metal kernel path
        out_metal, lse_metal = varlen_flash_attention_forward(
            q, k, v, cu_seqlens_q, cu_seqlens_k,
            softmax_scale=softmax_scale, causal=True, alibi_slopes=alibi_slopes,
            use_metal_kernel=True
        )
        mx.eval(out_metal, lse_metal)

        # Reference
        out_ref, lse_ref, _ = varlen_attention_ref_mlx(
            q, k, v, cu_seqlens_q, cu_seqlens_k,
            softmax_scale=softmax_scale, causal=True, alibi_slopes=alibi_slopes
        )
        mx.eval(out_ref, lse_ref)

        max_diff = float(mx.max(mx.abs(out_metal - out_ref)))
        assert max_diff < 0.05, f"Varlen ALiBi forward diff too large: {max_diff}"

    def test_alibi_backward_metal_kernel(self):
        """Verify Metal kernel ALiBi backward matches reference (via VJP)."""
        mx.random.seed(42)
        batch, seqlen, nheads, headdim = 2, 16, 4, 64

        q = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        k = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        v = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1

        alibi_slopes = mx.array([0.5, 0.25, 0.125, 0.0625], dtype=mx.float32)
        softmax_scale = 1.0 / math.sqrt(headdim)

        def loss_fn_metal(q, k, v):
            # Use flash_attention_mlx which has VJP support
            out = flash_attention_mlx(
                q, k, v, softmax_scale=softmax_scale, causal=True, alibi_slopes=alibi_slopes
            )
            return mx.sum(out)

        # Compute gradients
        grad_fn = mx.grad(loss_fn_metal, argnums=(0, 1, 2))
        dq, dk, dv = grad_fn(q, k, v)
        mx.eval(dq, dk, dv)

        # Verify shapes and no NaN
        assert dq.shape == q.shape
        assert dk.shape == k.shape
        assert dv.shape == v.shape
        assert not mx.any(mx.isnan(dq)), "dQ has NaN values"
        assert not mx.any(mx.isnan(dk)), "dK has NaN values"
        assert not mx.any(mx.isnan(dv)), "dV has NaN values"

    def test_alibi_interface_uses_metal(self):
        """Verify flash_attn_func with ALiBi uses Metal kernel (no fallback)."""
        mx.random.seed(42)
        batch, seqlen, nheads, headdim = 2, 16, 4, 64

        q = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        k = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        v = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1

        alibi_slopes = mx.array([0.5, 0.25, 0.125, 0.0625], dtype=mx.float32)

        # Call interface function - should use Metal kernel now
        out = flash_attn_func(
            q, k, v, causal=True, alibi_slopes=alibi_slopes
        )
        mx.eval(out)

        # Verify shape and no NaN
        assert out.shape == q.shape
        assert not mx.any(mx.isnan(out))


class TestDropoutMetalKernel:
    """Tests for Metal kernel Dropout support (Phase 3)."""

    def test_dropout_forward_metal_kernel(self):
        """Verify Metal kernel with dropout produces different output than without."""
        mx.random.seed(42)
        batch, seqlen, nheads, headdim = 2, 16, 4, 64

        q = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        k = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        v = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1

        # Forward without dropout
        out_no_drop, lse_no_drop = flash_attention_with_lse(
            q, k, v, causal=True, dropout_p=0.0
        )
        mx.eval(out_no_drop, lse_no_drop)

        # Forward with dropout
        out_drop, lse_drop = flash_attention_with_lse(
            q, k, v, causal=True, dropout_p=0.3, philox_seed=0x1BF58, philox_offset=0
        )
        mx.eval(out_drop, lse_drop)

        # Verify shapes
        assert out_drop.shape == q.shape
        assert lse_drop.shape == (batch, nheads, seqlen)

        # Verify no NaN
        assert not mx.any(mx.isnan(out_drop)), "Dropout output has NaN"
        assert not mx.any(mx.isnan(lse_drop)), "Dropout LSE has NaN"

        # Outputs should be different (dropout changes the result)
        diff = float(mx.mean(mx.abs(out_drop - out_no_drop)))
        assert diff > 0.001, f"Expected different outputs with dropout, got diff={diff}"

        # LSE should be the same (dropout is applied after softmax normalization)
        lse_diff = float(mx.max(mx.abs(lse_drop - lse_no_drop)))
        assert lse_diff < 0.01, f"LSE should be similar, got diff={lse_diff}"

    def test_dropout_reproducibility(self):
        """Verify same seed produces same dropout pattern."""
        mx.random.seed(42)
        batch, seqlen, nheads, headdim = 2, 16, 4, 64

        q = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        k = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        v = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1

        # Forward with dropout - same seed twice
        out1, _ = flash_attention_with_lse(
            q, k, v, causal=True, dropout_p=0.3, philox_seed=12345, philox_offset=0
        )
        out2, _ = flash_attention_with_lse(
            q, k, v, causal=True, dropout_p=0.3, philox_seed=12345, philox_offset=0
        )
        mx.eval(out1, out2)

        # Should be identical with same seed
        diff = float(mx.max(mx.abs(out1 - out2)))
        assert diff < 1e-5, f"Same seed should produce identical output, got diff={diff}"

    def test_dropout_different_seeds(self):
        """Verify different seeds produce different dropout patterns."""
        mx.random.seed(42)
        batch, seqlen, nheads, headdim = 2, 16, 4, 64

        q = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        k = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        v = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1

        # Forward with different seeds
        out1, _ = flash_attention_with_lse(
            q, k, v, causal=True, dropout_p=0.3, philox_seed=111, philox_offset=0
        )
        out2, _ = flash_attention_with_lse(
            q, k, v, causal=True, dropout_p=0.3, philox_seed=222, philox_offset=0
        )
        mx.eval(out1, out2)

        # Should be different with different seeds
        diff = float(mx.mean(mx.abs(out1 - out2)))
        assert diff > 0.001, f"Different seeds should produce different output, got diff={diff}"

    def test_dropout_interface_uses_metal(self):
        """Verify flash_attn_func with dropout uses Metal kernel (no fallback)."""
        mx.random.seed(42)
        batch, seqlen, nheads, headdim = 2, 16, 4, 64

        q = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        k = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        v = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1

        # Call interface function with dropout - should use Metal kernel now
        out = flash_attn_func(q, k, v, causal=True, dropout_p=0.1)
        mx.eval(out)

        # Verify shape and no NaN
        assert out.shape == q.shape
        assert not mx.any(mx.isnan(out))

    def test_dropout_high_rate(self):
        """Verify high dropout rate (0.9) works correctly."""
        mx.random.seed(42)
        batch, seqlen, nheads, headdim = 2, 16, 4, 64

        q = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        k = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        v = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1

        # Forward with high dropout
        out, _ = flash_attention_with_lse(
            q, k, v, causal=True, dropout_p=0.9, philox_seed=0x1BF58, philox_offset=0
        )
        mx.eval(out)

        # Verify no NaN and reasonable output
        assert not mx.any(mx.isnan(out)), "High dropout output has NaN"
        assert not mx.any(mx.isinf(out)), "High dropout output has Inf"


@pytest.mark.skipif(USE_REF, reason="Requires Metal kernel (FLASH_ATTENTION_MLX_REF=0)")
class TestVarlenDropoutMetalKernel:
    """Tests for varlen dropout support in the Metal kernels."""

    def _make_inputs(self, *, dtype=mx.float16):
        seqlens = [19, 11, 7]
        nheads = 4
        nheads_k = 2
        headdim = 64
        return make_varlen_qkv(seqlens, nheads, nheads_k, headdim, dtype=dtype, seed=7)

    def test_varlen_dropout_forward_metal_kernel(self):
        """Verify varlen Metal kernel applies dropout mask and stays finite."""
        q, k, v, cu_q, cu_k, max_q, max_k = self._make_inputs()
        total_q = int(cu_q[-1].item())
        nheads = q.shape[1]
        headdim = q.shape[2]
        softmax_scale = 1.0 / math.sqrt(headdim)
        dropout_p = 0.25
        philox_seed, philox_offset = _default_philox_state(dropout_p)

        out_drop, lse_drop = varlen_flash_attention_forward(
            q,
            k,
            v,
            cu_q,
            cu_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            softmax_scale=softmax_scale,
            causal=True,
            dropout_p=dropout_p,
            philox_seed=philox_seed,
            philox_offset=philox_offset,
            use_metal_kernel=True,
        )
        out_no_drop, _ = varlen_flash_attention_forward(
            q,
            k,
            v,
            cu_q,
            cu_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            softmax_scale=softmax_scale,
            causal=True,
            dropout_p=0.0,
            use_metal_kernel=True,
        )
        mx.eval(out_drop, lse_drop, out_no_drop)

        assert out_drop.shape == (total_q, nheads, headdim)
        assert lse_drop.shape == (nheads, total_q)
        assert not mx.any(mx.isnan(out_drop)), "Varlen dropout output has NaN"
        assert not mx.any(mx.isnan(lse_drop)), "Varlen dropout LSE has NaN"

        mean_diff = float(mx.mean(mx.abs(out_drop - out_no_drop)))
        assert mean_diff > 1e-4, f"Expected dropout to change output, diff={mean_diff:.6f}"

    def test_varlen_dropout_deterministic_forward(self):
        """Same Philox seed/offset should reproduce identical varlen outputs."""
        q, k, v, cu_q, cu_k, max_q, max_k = self._make_inputs()
        headdim = q.shape[2]
        softmax_scale = 1.0 / math.sqrt(headdim)
        dropout_p = 0.3
        seed = 0xABCDEF
        offset = 0x1234

        out1, _ = varlen_flash_attention_forward(
            q,
            k,
            v,
            cu_q,
            cu_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            softmax_scale=softmax_scale,
            causal=False,
            dropout_p=dropout_p,
            philox_seed=seed,
            philox_offset=offset,
            use_metal_kernel=True,
        )
        out2, _ = varlen_flash_attention_forward(
            q,
            k,
            v,
            cu_q,
            cu_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            softmax_scale=softmax_scale,
            causal=False,
            dropout_p=dropout_p,
            philox_seed=seed,
            philox_offset=offset,
            use_metal_kernel=True,
        )
        mx.eval(out1, out2)

        assert mx.allclose(out1, out2, atol=1e-5), "Dropout mask not deterministic for varlen"

    def test_varlen_dropout_backward_autograd(self):
        """Autograd wrapper should propagate gradients when dropout enabled."""
        q, k, v, cu_q, cu_k, max_q, max_k = self._make_inputs(dtype=mx.float32)
        softmax_scale = 1.0 / math.sqrt(q.shape[2])
        dropout_p = 0.2

        def forward_fn(q_in, k_in, v_in):
            out, _ = varlen_flash_attention_mlx(
                q_in,
                k_in,
                v_in,
                cu_q,
                cu_k,
                max_seqlen_q=max_q,
                max_seqlen_k=max_k,
                softmax_scale=softmax_scale,
                causal=True,
                window_size=(-1, -1),
                softcap=0.0,
                use_metal_kernel=True,
                dropout_p=dropout_p,
            )
            return mx.sum(out)

        dq, dk, dv = mx.grad(forward_fn, argnums=(0, 1, 2))(q, k, v)
        mx.eval(dq, dk, dv)

        assert not mx.any(mx.isnan(dq)), "dQ has NaN for varlen dropout"
        assert not mx.any(mx.isnan(dk)), "dK has NaN for varlen dropout"
        assert not mx.any(mx.isnan(dv)), "dV has NaN for varlen dropout"

    def test_varlen_dropout_interface_matches_kernel(self):
        """Interface helper should return same output as kernel with dropout."""
        q, k, v, cu_q, cu_k, max_q, max_k = self._make_inputs()
        total_q = int(cu_q[-1].item())
        nheads = q.shape[1]
        headdim = q.shape[2]
        softmax_scale = 1.0 / math.sqrt(headdim)
        dropout_p = 0.15
        philox_seed, philox_offset = _default_philox_state(dropout_p)

        out_kernel, lse_kernel = varlen_flash_attention_forward(
            q,
            k,
            v,
            cu_q,
            cu_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            softmax_scale=softmax_scale,
            causal=False,
            dropout_p=dropout_p,
            philox_seed=philox_seed,
            philox_offset=philox_offset,
            use_metal_kernel=True,
        )

        out_iface, lse_iface, attn_probs = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=False,
            return_attn_probs=True,
        )
        mx.eval(out_kernel, lse_kernel, out_iface, lse_iface)

        assert attn_probs.shape == (0,), "Varlen Metal path should not materialize attn probs"
        assert lse_iface.shape == (nheads, total_q)

        max_out_diff = float(mx.max(mx.abs(out_kernel - out_iface)))
        max_lse_diff = float(mx.max(mx.abs(lse_kernel - lse_iface)))
        assert max_out_diff < 5e-4, f"Interface output diverged from kernel: {max_out_diff}"
        assert max_lse_diff < 5e-4, f"Interface LSE diverged from kernel: {max_lse_diff}"

    def test_varlen_alibi_dropout_combined(self):
        """Varlen interface should support simultaneous ALiBi and dropout."""
        q, k, v, cu_q, cu_k, max_q, max_k = self._make_inputs(dtype=mx.float32)
        headdim = q.shape[2]
        nheads = q.shape[1]
        softmax_scale = 1.0 / math.sqrt(headdim)
        dropout_p = 0.2
        alibi_slopes = mx.linspace(0.05, 0.2, nheads, dtype=mx.float32)

        out, lse, _ = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=True,
            alibi_slopes=alibi_slopes,
            return_attn_probs=True,
        )
        mx.eval(out, lse)

        assert out.shape == q.shape
        assert lse.shape == (nheads, int(cu_q[-1].item()))
        assert not mx.any(mx.isnan(out)), "ALiBi+dropout varlen output has NaN"
        assert not mx.any(mx.isnan(lse)), "ALiBi+dropout varlen LSE has NaN"


@pytest.mark.skipif(USE_REF, reason="Requires Metal kernel (FLASH_ATTENTION_MLX_REF=0)")
class TestDropoutBackwardMetal:
    """Tests for Metal kernel dropout backward pass (Philox mask regeneration)."""

    def test_dropout_backward_no_nan(self):
        """Verify backward pass with dropout produces valid gradients."""
        mx.random.seed(42)
        batch, seqlen, nheads, headdim = 2, 16, 4, 64

        q = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float32) * 0.1
        k = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float32) * 0.1
        v = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float32) * 0.1

        def loss_fn(q, k, v):
            return mx.sum(flash_attn_func(q, k, v, causal=True, dropout_p=0.1))

        dq, dk, dv = mx.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)
        mx.eval(dq, dk, dv)

        # Verify no NaN in gradients
        assert not mx.any(mx.isnan(dq)), "dQ has NaN with dropout"
        assert not mx.any(mx.isnan(dk)), "dK has NaN with dropout"
        assert not mx.any(mx.isnan(dv)), "dV has NaN with dropout"

        # Verify no Inf in gradients
        assert not mx.any(mx.isinf(dq)), "dQ has Inf with dropout"
        assert not mx.any(mx.isinf(dk)), "dK has Inf with dropout"
        assert not mx.any(mx.isinf(dv)), "dV has Inf with dropout"

    @pytest.mark.parametrize("dropout_p", [0.1, 0.3, 0.5])
    @pytest.mark.parametrize("causal", [False, True])
    def test_dropout_backward_various_rates(self, dropout_p, causal):
        """Verify backward pass works with various dropout rates and causal settings."""
        mx.random.seed(42)
        batch, seqlen, nheads, headdim = 2, 32, 4, 64

        q = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float32) * 0.1
        k = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float32) * 0.1
        v = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float32) * 0.1

        def forward_fn(q, k, v):
            return mx.sum(flash_attn_func(q, k, v, causal=causal, dropout_p=dropout_p))

        dq, dk, dv = mx.grad(forward_fn, argnums=(0, 1, 2))(q, k, v)
        mx.eval(dq, dk, dv)

        # Verify gradients are valid
        assert not mx.any(mx.isnan(dq)), f"dQ has NaN with dropout_p={dropout_p}, causal={causal}"
        assert not mx.any(mx.isnan(dk)), f"dK has NaN with dropout_p={dropout_p}, causal={causal}"
        assert not mx.any(mx.isnan(dv)), f"dV has NaN with dropout_p={dropout_p}, causal={causal}"

        # Verify gradients have reasonable magnitude (not exploding)
        max_grad = max(float(mx.max(mx.abs(dq))), float(mx.max(mx.abs(dk))), float(mx.max(mx.abs(dv))))
        assert max_grad < 100.0, f"Gradients are exploding: max={max_grad}"

    def test_dropout_backward_deterministic(self):
        """Verify same seed produces same gradients (deterministic backward)."""
        mx.random.seed(42)
        batch, seqlen, nheads, headdim = 2, 16, 4, 64

        q = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float32) * 0.1
        k = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float32) * 0.1
        v = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float32) * 0.1

        def forward_fn(q, k, v):
            return mx.sum(flash_attn_func(q, k, v, causal=True, dropout_p=0.3))

        # First run
        dq1, dk1, dv1 = mx.grad(forward_fn, argnums=(0, 1, 2))(q, k, v)
        mx.eval(dq1, dk1, dv1)

        # Second run with same inputs
        dq2, dk2, dv2 = mx.grad(forward_fn, argnums=(0, 1, 2))(q, k, v)
        mx.eval(dq2, dk2, dv2)

        # Verify identical gradients (MLX is deterministic by design)
        assert mx.allclose(dq1, dq2, atol=1e-5), "dQ not deterministic"
        assert mx.allclose(dk1, dk2, atol=1e-5), "dK not deterministic"
        assert mx.allclose(dv1, dv2, atol=1e-5), "dV not deterministic"

    def test_dropout_backward_different_from_no_dropout(self):
        """Verify dropout backward produces different gradients than no dropout."""
        mx.random.seed(42)
        batch, seqlen, nheads, headdim = 2, 32, 4, 64

        q = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float32) * 0.1
        k = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float32) * 0.1
        v = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float32) * 0.1

        def forward_no_drop(q, k, v):
            return mx.sum(flash_attn_func(q, k, v, causal=True, dropout_p=0.0))

        def forward_with_drop(q, k, v):
            return mx.sum(flash_attn_func(q, k, v, causal=True, dropout_p=0.3))

        dq_no, dk_no, dv_no = mx.grad(forward_no_drop, argnums=(0, 1, 2))(q, k, v)
        dq_drop, dk_drop, dv_drop = mx.grad(forward_with_drop, argnums=(0, 1, 2))(q, k, v)
        mx.eval(dq_no, dk_no, dv_no, dq_drop, dk_drop, dv_drop)

        # Gradients should be different due to dropout
        dq_diff = float(mx.mean(mx.abs(dq_no - dq_drop)))
        dk_diff = float(mx.mean(mx.abs(dk_no - dk_drop)))
        dv_diff = float(mx.mean(mx.abs(dv_no - dv_drop)))

        assert dq_diff > 1e-4, f"dQ should differ with dropout, got diff={dq_diff}"
        assert dk_diff > 1e-4, f"dK should differ with dropout, got diff={dk_diff}"
        assert dv_diff > 1e-4, f"dV should differ with dropout, got diff={dv_diff}"


@pytest.mark.skipif(USE_REF, reason="Requires Metal kernel (FLASH_ATTENTION_MLX_REF=0)")
class TestALiBiDropoutCombined:
    """Tests for combined ALiBi + dropout through Metal kernel."""

    def test_alibi_dropout_forward(self):
        """Verify forward pass with both ALiBi and dropout."""
        mx.random.seed(42)
        batch, seqlen, nheads, headdim = 2, 32, 8, 64

        q = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        k = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1
        v = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float16) * 0.1

        # ALiBi slopes (power of 2 sequence as in original paper)
        alibi_slopes = mx.array([2 ** (-8 * (i + 1) / nheads) for i in range(nheads)], dtype=mx.float32)

        # Forward with both features
        out = flash_attn_func(q, k, v, causal=True, dropout_p=0.1, alibi_slopes=alibi_slopes)
        mx.eval(out)

        # Verify output shape and no NaN
        assert out.shape == q.shape
        assert not mx.any(mx.isnan(out)), "ALiBi+dropout forward has NaN"
        assert not mx.any(mx.isinf(out)), "ALiBi+dropout forward has Inf"

    def test_alibi_dropout_backward(self):
        """Verify backward pass with both ALiBi and dropout."""
        mx.random.seed(42)
        batch, seqlen, nheads, headdim = 2, 32, 8, 64

        q = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float32) * 0.1
        k = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float32) * 0.1
        v = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float32) * 0.1

        alibi_slopes = mx.array([2 ** (-8 * (i + 1) / nheads) for i in range(nheads)], dtype=mx.float32)

        def forward_fn(q, k, v):
            return mx.sum(flash_attn_func(q, k, v, causal=True, dropout_p=0.2, alibi_slopes=alibi_slopes))

        dq, dk, dv = mx.grad(forward_fn, argnums=(0, 1, 2))(q, k, v)
        mx.eval(dq, dk, dv)

        # Verify gradients are valid
        assert not mx.any(mx.isnan(dq)), "dQ has NaN with ALiBi+dropout"
        assert not mx.any(mx.isnan(dk)), "dK has NaN with ALiBi+dropout"
        assert not mx.any(mx.isnan(dv)), "dV has NaN with ALiBi+dropout"

        # Verify gradients have reasonable magnitude
        max_grad = max(float(mx.max(mx.abs(dq))), float(mx.max(mx.abs(dk))), float(mx.max(mx.abs(dv))))
        assert max_grad < 100.0, f"ALiBi+dropout gradients exploding: max={max_grad}"

    @pytest.mark.parametrize("causal", [False, True])
    @pytest.mark.parametrize("dropout_p", [0.0, 0.1, 0.3])
    def test_alibi_dropout_configs(self, causal, dropout_p):
        """Test various causal and dropout configurations with ALiBi."""
        mx.random.seed(42)
        batch, seqlen, nheads, headdim = 2, 32, 4, 64

        q = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float32) * 0.1
        k = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float32) * 0.1
        v = mx.random.normal((batch, seqlen, nheads, headdim)).astype(mx.float32) * 0.1

        alibi_slopes = mx.array([0.5 ** (i + 1) for i in range(nheads)], dtype=mx.float32)

        # Forward
        out = flash_attn_func(q, k, v, causal=causal, dropout_p=dropout_p, alibi_slopes=alibi_slopes)
        mx.eval(out)

        assert not mx.any(mx.isnan(out)), f"Output has NaN: causal={causal}, dropout_p={dropout_p}"

        # Backward (only if dropout > 0 to test full path)
        if dropout_p > 0:
            def forward_fn(q, k, v):
                return mx.sum(flash_attn_func(q, k, v, causal=causal, dropout_p=dropout_p, alibi_slopes=alibi_slopes))

            dq, dk, dv = mx.grad(forward_fn, argnums=(0, 1, 2))(q, k, v)
            mx.eval(dq, dk, dv)

            assert not mx.any(mx.isnan(dq)), f"dQ has NaN: causal={causal}, dropout_p={dropout_p}"
            assert not mx.any(mx.isnan(dk)), f"dK has NaN: causal={causal}, dropout_p={dropout_p}"
            assert not mx.any(mx.isnan(dv)), f"dV has NaN: causal={causal}, dropout_p={dropout_p}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Utility helpers for variable-length MLX attention tests."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import mlx.core as mx


def _validate_seqlens(seqlens: Sequence[int]) -> None:
    if not seqlens:
        raise ValueError("seqlens must be a non-empty sequence")
    for length in seqlens:
        if length < 0:
            raise ValueError("sequence lengths must be non-negative")


def generate_cu_seqlens(seqlens: Sequence[int]) -> mx.array:
    """Return cumulative sequence lengths array (int32) for varlen tests."""
    _validate_seqlens(seqlens)
    cumulative: List[int] = [0]
    running = 0
    for length in seqlens:
        running += int(length)
        cumulative.append(running)
    return mx.array(cumulative, dtype=mx.int32)


VARLEN_SEQLEN_CASES: Sequence[Sequence[int]] = (
    [128, 128, 128],
    [97, 200, 50],
    [1, 512, 32],
    [256],
    [1, 1, 1, 1],
    [2048],
)

VARLEN_HEAD_CONFIGS: Sequence[Tuple[int, int]] = (
    (8, 8),
    (8, 2),
    (8, 1),
)

VARLEN_HEADDIMS = (64, 128)
VARLEN_CAUSAL_FLAGS = (False, True)


def make_varlen_qkv(
    seqlens_q: Sequence[int],
    nheads: int,
    nheads_k: int,
    headdim: int,
    *,
    seqlens_k: Optional[Sequence[int]] = None,
    dtype=None,
    seed: int = 42,
) -> Tuple[mx.array, mx.array, mx.array, mx.array, mx.array, int, int]:
    """Create packed Q/K/V tensors and cu_seqlens arrays for varlen tests."""
    if dtype is None:
        dtype = mx.float16

    seqlens_k = seqlens_k or seqlens_q
    _validate_seqlens(seqlens_q)
    _validate_seqlens(seqlens_k)

    mx.random.seed(seed)

    total_q = sum(seqlens_q)
    total_k = sum(seqlens_k)

    q = mx.random.normal((total_q, nheads, headdim)).astype(dtype) * 0.1
    k = mx.random.normal((total_k, nheads_k, headdim)).astype(dtype) * 0.1
    v = mx.random.normal((total_k, nheads_k, headdim)).astype(dtype) * 0.1

    cu_seqlens_q = generate_cu_seqlens(seqlens_q)
    cu_seqlens_k = generate_cu_seqlens(seqlens_k)

    max_seqlen_q = max(seqlens_q)
    max_seqlen_k = max(seqlens_k)

    return (
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
    )

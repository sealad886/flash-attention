"""MLX Flash Attention utilities and configuration helpers."""

import math
import os
from typing import Optional, Literal, Tuple, Union

import mlx.core as mx

# Environment variable configuration
DEBUG = os.environ.get('FLASH_ATTENTION_MLX_DEBUG', '0').lower() in ('1', 'true', 'yes')
USE_REF = os.environ.get('FLASH_ATTENTION_MLX_REF', '1').lower() in ('1', 'true', 'yes')
PERF = os.environ.get('FLASH_ATTENTION_MLX_PERF', '0').lower() in ('1', 'true', 'yes')

# Supported head dimensions
SUPPORTED_HEAD_DIMS = [32, 64, 96, 128, 160, 192, 256]


def log2_if_power_of_two(value: int) -> int:
    """Return log2(value) if value is a power of two, otherwise -1."""
    if value <= 0 or (value & (value - 1)) != 0:
        return -1
    return int(math.log2(value))


def maybe_contiguous(x: mx.array) -> mx.array:
    """
    Ensure array is contiguous in memory.

    MLX arrays are typically already contiguous due to the unified memory model,
    but this function provides a consistent interface.
    """
    # MLX doesn't have an is_contiguous check like PyTorch, but arrays
    # created from operations are typically contiguous. For safety,
    # we can use mx.array to ensure a fresh copy if needed.
    return x


def check_args(q: mx.array, k: mx.array, v: mx.array) -> None:
    """
    Validate input tensor shapes and types.

    Args:
        q: Query tensor of shape (batch, seqlen_q, nheads, headdim)
        k: Key tensor of shape (batch, seqlen_k, nheads_k, headdim)
        v: Value tensor of shape (batch, seqlen_k, nheads_k, headdim)

    Raises:
        ValueError: If tensor shapes or types are invalid
    """
    if q.ndim != 4:
        raise ValueError(f"Query tensor must be 4D (batch, seqlen, nheads, headdim), got {q.ndim}D")
    if k.ndim != 4:
        raise ValueError(f"Key tensor must be 4D (batch, seqlen, nheads, headdim), got {k.ndim}D")
    if v.ndim != 4:
        raise ValueError(f"Value tensor must be 4D (batch, seqlen, nheads, headdim), got {v.ndim}D")

    batch_q, seqlen_q, nheads_q, headdim_q = q.shape
    batch_k, seqlen_k, nheads_k, headdim_k = k.shape
    batch_v, seqlen_v, nheads_v, headdim_v = v.shape

    if batch_q != batch_k or batch_q != batch_v:
        raise ValueError(f"Batch sizes must match: q={batch_q}, k={batch_k}, v={batch_v}")

    if seqlen_k != seqlen_v:
        raise ValueError(f"Key and value sequence lengths must match: k={seqlen_k}, v={seqlen_v}")

    if nheads_k != nheads_v:
        raise ValueError(f"Key and value head counts must match: k={nheads_k}, v={nheads_v}")

    if nheads_q % nheads_k != 0:
        raise ValueError(
            f"Query heads ({nheads_q}) must be divisible by key/value heads ({nheads_k}) for GQA/MQA"
        )

    if headdim_q != headdim_k or headdim_q != headdim_v:
        raise ValueError(f"Head dimensions must match: q={headdim_q}, k={headdim_k}, v={headdim_v}")

    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(f"Data types must match: q={q.dtype}, k={k.dtype}, v={v.dtype}")


def validate_paged_kv_params(
    k_cache: mx.array,
    v_cache: mx.array,
    page_table: mx.array,
    batch_size: int,
) -> Tuple[int, int]:
    """Validate paged KV cache arguments and return (page_size, max_pages_per_seq)."""
    if page_table.dtype != mx.int32:
        raise TypeError(f"page_table must be int32, got {page_table.dtype}")
    if page_table.ndim != 2:
        raise ValueError(f"page_table must be 2D, got {page_table.ndim}D")
    if page_table.shape[0] != batch_size:
        raise ValueError(
            f"page_table batch dimension ({page_table.shape[0]}) must match queries ({batch_size})",
        )

    if k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError("Paged KV caches must be 4D tensors (num_pages, page_size, nheads_k, headdim)")
    if k_cache.shape != v_cache.shape:
        raise ValueError(
            "K/V cache tensors must share the same shape for paged KV attention",
        )

    page_size = int(k_cache.shape[1])
    max_pages_per_seq = int(page_table.shape[1])
    num_pages = int(k_cache.shape[0])

    if page_size <= 0:
        raise ValueError("page_size inferred from k_cache must be > 0")
    if max_pages_per_seq <= 0:
        raise ValueError("page_table must contain at least one page per sequence")
    if num_pages <= 0:
        raise ValueError("k_cache must contain at least one physical page")

    # Ensure block table entries never reference pages outside the physical cache range.
    safe_entries = mx.where(page_table < 0, mx.zeros_like(page_table), page_table)
    max_entry = int(mx.max(safe_entries).item())
    if max_entry >= num_pages:
        raise ValueError(
            f"page_table references page index {max_entry}, but k_cache only has {num_pages} pages",
        )

    return page_size, max_pages_per_seq


def check_varlen_args(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    cu_seqlens_q: mx.array,
    cu_seqlens_k: mx.array,
    max_seqlen_q: int,
    max_seqlen_k: int,
) -> None:
    """
    Validate variable-length input tensor shapes and types.

    Args:
        q: Query tensor of shape (total_q, nheads, headdim)
        k: Key tensor of shape (total_k, nheads_k, headdim)
        v: Value tensor of shape (total_k, nheads_k, headdim)
        cu_seqlens_q: Cumulative sequence lengths for queries
        cu_seqlens_k: Cumulative sequence lengths for keys
        max_seqlen_q: Maximum query sequence length
        max_seqlen_k: Maximum key sequence length
    """
    if q.ndim != 3:
        raise ValueError(f"Query tensor must be 3D (total, nheads, headdim), got {q.ndim}D")
    if k.ndim != 3:
        raise ValueError(f"Key tensor must be 3D (total, nheads, headdim), got {k.ndim}D")
    if v.ndim != 3:
        raise ValueError(f"Value tensor must be 3D (total, nheads, headdim), got {v.ndim}D")

    if cu_seqlens_q.ndim != 1:
        raise ValueError(f"cu_seqlens_q must be 1D, got {cu_seqlens_q.ndim}D")
    if cu_seqlens_k.ndim != 1:
        raise ValueError(f"cu_seqlens_k must be 1D, got {cu_seqlens_k.ndim}D")

    if len(cu_seqlens_q) != len(cu_seqlens_k):
        raise ValueError(
            f"cu_seqlens_q and cu_seqlens_k must have same length: "
            f"{len(cu_seqlens_q)} vs {len(cu_seqlens_k)}"
        )

    if len(cu_seqlens_q) < 2:
        raise ValueError("cu_seqlens must have at least 2 elements (batch + 1)")


def get_shapes_from_layout(
    q: mx.array,
    k: mx.array,
    layout: Literal["bshd", "bhsd", "thd"],
    cu_seqlens_q: Optional[mx.array] = None,
    cu_seqlens_k: Optional[mx.array] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
) -> tuple:
    """
    Extract shape information from tensors based on layout.

    Args:
        q: Query tensor
        k: Key tensor
        layout: Tensor layout ("bshd", "bhsd", or "thd" for variable length)
        cu_seqlens_q: Cumulative sequence lengths for queries (for "thd")
        cu_seqlens_k: Cumulative sequence lengths for keys (for "thd")
        max_seqlen_q: Maximum query sequence length (for "thd")
        max_seqlen_k: Maximum key sequence length (for "thd")

    Returns:
        Tuple of (batch, nheads_q, nheads_k, head_size, seqlen_q, seqlen_k)
    """
    if layout == "bshd":
        batch, seqlen_q, nheads_q, head_size = q.shape
        _, seqlen_k, nheads_k, _ = k.shape
    elif layout == "bhsd":
        batch, nheads_q, seqlen_q, head_size = q.shape
        _, nheads_k, seqlen_k, _ = k.shape
    elif layout == "thd":
        if cu_seqlens_q is None or cu_seqlens_k is None:
            raise ValueError("cu_seqlens required for 'thd' layout")
        batch = len(cu_seqlens_q) - 1
        _, nheads_q, head_size = q.shape
        _, nheads_k, _ = k.shape
        seqlen_q = max_seqlen_q if max_seqlen_q is not None else 0
        seqlen_k = max_seqlen_k if max_seqlen_k is not None else 0
    else:
        raise ValueError(f"Unknown layout: {layout}")

    return batch, nheads_q, nheads_k, head_size, seqlen_q, seqlen_k


class MetaData:
    """
    Metadata container for attention computation parameters.

    Similar to the MetaData class in the Triton AMD backend, this class
    holds configuration and computed values needed for attention.
    """
    cu_seqlens_q: Optional[mx.array] = None
    cu_seqlens_k: Optional[mx.array] = None
    max_seqlens_q: int = 0
    max_seqlens_k: int = 0
    alibi_slopes: Optional[mx.array] = None
    causal: bool = False
    varlen: bool = False
    layout: Optional[Literal["bshd", "bhsd", "thd"]] = None
    cache_seqlens: Optional[Union[int, mx.array]] = None
    cache_batch_idx: Optional[mx.array] = None
    return_scores: bool = False
    dropout_p: float = 0.0
    philox_seed: Optional[int] = None
    philox_offset: Optional[int] = None
    softcap: float = 0.0
    window_size_left: int = -1
    window_size_right: int = -1

    def __init__(self, sm_scale: float = 1.0):
        self.sm_scale = sm_scale

    def __repr__(self) -> str:
        return (
            f"MetaData(\n"
            f"  sm_scale={self.sm_scale},\n"
            f"  cu_seqlens_q={self.cu_seqlens_q},\n"
            f"  cu_seqlens_k={self.cu_seqlens_k},\n"
            f"  max_seqlens_q={self.max_seqlens_q},\n"
            f"  max_seqlens_k={self.max_seqlens_k},\n"
            f"  alibi_slopes={self.alibi_slopes},\n"
            f"  causal={self.causal},\n"
            f"  varlen={self.varlen},\n"
            f"  layout={self.layout},\n"
            f"  dropout_p={self.dropout_p},\n"
            f"  softcap={self.softcap},\n"
            f"  window_size=({self.window_size_left}, {self.window_size_right})\n"
            f")"
        )

    def set_varlen_params(
        self,
        cu_seqlens_q: mx.array,
        cu_seqlens_k: mx.array,
        max_seqlen_q: int,
        max_seqlen_k: int,
    ) -> None:
        """Configure for variable-length sequences."""
        self.varlen = True
        self.layout = "thd"
        self.cu_seqlens_q = cu_seqlens_q
        self.cu_seqlens_k = cu_seqlens_k
        self.max_seqlens_q = max_seqlen_q
        self.max_seqlens_k = max_seqlen_k

    def need_causal(self, causal: bool) -> None:
        """Set causal masking mode."""
        self.causal = causal

    def need_alibi(self, alibi_slopes: mx.array, batch: int, nheads: int) -> None:
        """Configure ALiBi attention bias."""
        if alibi_slopes.ndim == 1:
            if alibi_slopes.shape[0] != nheads:
                raise ValueError(
                    f"ALiBi slopes must have {nheads} elements, got {alibi_slopes.shape[0]}"
                )
        elif alibi_slopes.ndim == 2:
            if alibi_slopes.shape != (batch, nheads):
                raise ValueError(
                    f"ALiBi slopes must have shape ({batch}, {nheads}), "
                    f"got {alibi_slopes.shape}"
                )
        else:
            raise ValueError(f"ALiBi slopes must be 1D or 2D, got {alibi_slopes.ndim}D")
        self.alibi_slopes = alibi_slopes

    def need_dropout(self, dropout_p: float, return_softmax: bool = True) -> None:
        """Configure dropout parameters."""
        self.dropout_p = dropout_p
        self.return_scores = return_softmax
        # Fixed seed for reproducibility in testing
        self.philox_seed = 0x1BF58
        self.philox_offset = 0x1D4B49

    def need_window(self, window_size_left: int, window_size_right: int) -> None:
        """Configure sliding window attention."""
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right

    def need_softcap(self, softcap: float) -> None:
        """Configure attention logit soft capping."""
        self.softcap = softcap

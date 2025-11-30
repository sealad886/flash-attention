"""
MLX Flash Attention Parameters

This module defines dataclasses for attention parameters, mirroring the
CUDA Flash_fwd_params and Flash_bwd_params structures for compatibility
and consistency across backends.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import mlx.core as mx


@dataclass
class AttentionParams:
    """
    Parameters for Flash Attention forward pass.

    This dataclass mirrors the CUDA Flash_fwd_params struct, adapted for MLX.
    It contains all the information needed to execute attention computation.

    Attributes:
        q: Query tensor of shape (batch, seqlen_q, nheads, headdim)
        k: Key tensor of shape (batch, seqlen_k, nheads_k, headdim)
        v: Value tensor of shape (batch, seqlen_k, nheads_k, headdim)
        out: Pre-allocated output tensor (optional)
        softmax_lse: Pre-allocated log-sum-exp output (optional)

        # Dimensions
        batch: Batch size
        seqlen_q: Query sequence length
        seqlen_k: Key/value sequence length
        nheads: Number of query heads
        nheads_k: Number of key/value heads (for GQA/MQA)
        headdim: Head dimension

        # Scaling
        softmax_scale: Scaling factor for attention (typically 1/sqrt(d))
        softmax_scale_log2: log2(softmax_scale) for use with exp2

        # Masking
        causal: Whether to apply causal masking
        window_size_left: Left window size for local attention (-1 = full)
        window_size_right: Right window size for local attention (-1 = full)
        softcap: Soft cap for attention logits (0 = disabled)

        # Variable length
        cu_seqlens_q: Cumulative sequence lengths for queries
        cu_seqlens_k: Cumulative sequence lengths for keys
        max_seqlen_q: Maximum query sequence length (for varlen)
        max_seqlen_k: Maximum key sequence length (for varlen)

        # Dropout
        dropout_p: Dropout probability (0 = disabled)
        philox_seed: RNG seed for dropout
        philox_offset: RNG offset for dropout

        # ALiBi
        alibi_slopes: ALiBi slopes tensor

        # KV Cache
        k_cache: Cached keys tensor
        v_cache: Cached values tensor
        cache_seqlens: Current cache sequence lengths
        cache_batch_idx: Batch indices for cache access

        # Misc
        is_bf16: Whether using bfloat16
        return_softmax: Whether to return softmax probabilities
    """

    # Input tensors
    q: mx.array
    k: mx.array
    v: mx.array
    out: Optional[mx.array] = None
    softmax_lse: Optional[mx.array] = None

    # Dimensions (computed from inputs if not provided)
    batch: int = 0
    seqlen_q: int = 0
    seqlen_k: int = 0
    nheads: int = 0
    nheads_k: int = 0
    headdim: int = 0

    # Scaling
    softmax_scale: float = 0.0
    softmax_scale_log2: float = 0.0

    # Masking
    causal: bool = False
    window_size_left: int = -1
    window_size_right: int = -1
    softcap: float = 0.0

    # Variable length sequences
    cu_seqlens_q: Optional[mx.array] = None
    cu_seqlens_k: Optional[mx.array] = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    is_varlen: bool = False

    # Dropout
    dropout_p: float = 0.0
    philox_seed: int = 0
    philox_offset: int = 0

    # ALiBi
    alibi_slopes: Optional[mx.array] = None

    # KV Cache
    k_cache: Optional[mx.array] = None
    v_cache: Optional[mx.array] = None
    cache_seqlens: Optional[mx.array] = None
    cache_batch_idx: Optional[mx.array] = None

    # Flags
    is_bf16: bool = False
    return_softmax: bool = False
    deterministic: bool = False

    def __post_init__(self):
        """Compute derived values from input tensors."""
        if self.q is not None:
            if self.q.ndim == 4:
                # Standard layout: (batch, seqlen, nheads, headdim)
                self.batch = self.q.shape[0]
                self.seqlen_q = self.q.shape[1]
                self.nheads = self.q.shape[2]
                self.headdim = self.q.shape[3]
            elif self.q.ndim == 3:
                # Variable length layout: (total, nheads, headdim)
                self.is_varlen = True
                self.nheads = self.q.shape[1]
                self.headdim = self.q.shape[2]

        if self.k is not None:
            if self.k.ndim == 4:
                self.seqlen_k = self.k.shape[1]
                self.nheads_k = self.k.shape[2]
            elif self.k.ndim == 3:
                self.nheads_k = self.k.shape[1]

        # Compute scale if not provided
        if self.softmax_scale == 0.0 and self.headdim > 0:
            import math
            self.softmax_scale = 1.0 / math.sqrt(self.headdim)
            self.softmax_scale_log2 = math.log2(self.softmax_scale)

        # Detect dtype
        if self.q is not None:
            self.is_bf16 = self.q.dtype == mx.bfloat16


@dataclass
class AttentionBackwardParams(AttentionParams):
    """
    Parameters for Flash Attention backward pass.

    Extends AttentionParams with gradient tensors and backward-specific fields.
    """

    # Output from forward (needed for backward)
    o: Optional[mx.array] = None

    # Gradient inputs
    do: Optional[mx.array] = None

    # Gradient outputs (pre-allocated or computed)
    dq: Optional[mx.array] = None
    dk: Optional[mx.array] = None
    dv: Optional[mx.array] = None

    # Accumulator for atomics-free gradient computation
    dq_accum: Optional[mx.array] = None
    dk_accum: Optional[mx.array] = None
    dv_accum: Optional[mx.array] = None

    # Softmax sum for backward
    dsoftmax_sum: Optional[mx.array] = None

    # Deterministic mode (may be slower but reproducible)
    deterministic: bool = False


def create_attention_params(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[mx.array] = None,
    return_softmax: bool = False,
    deterministic: bool = False,
) -> AttentionParams:
    """
    Factory function to create AttentionParams from user inputs.

    Args:
        q: Query tensor
        k: Key tensor
        v: Value tensor
        dropout_p: Dropout probability
        softmax_scale: Scaling factor (default: 1/sqrt(headdim))
        causal: Whether to apply causal masking
        window_size: (left, right) window sizes for local attention
        softcap: Soft cap for attention logits
        alibi_slopes: ALiBi slopes
        return_softmax: Whether to return softmax probabilities
        deterministic: Whether to use deterministic mode

    Returns:
        Configured AttentionParams instance
    """
    import math

    headdim = q.shape[-1]
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(headdim)

    params = AttentionParams(
        q=q,
        k=k,
        v=v,
        softmax_scale=softmax_scale,
        softmax_scale_log2=math.log2(softmax_scale) if softmax_scale > 0 else 0.0,
        causal=causal,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
        softcap=softcap,
        dropout_p=dropout_p,
        alibi_slopes=alibi_slopes,
        return_softmax=return_softmax,
        deterministic=deterministic,
    )

    # Set dropout RNG state if needed
    if dropout_p > 0.0:
        params.philox_seed = 0x1BF58
        params.philox_offset = 0x1D4B49

    return params


def create_backward_params(
    fwd_params: AttentionParams,
    o: mx.array,
    do: mx.array,
    dq: Optional[mx.array] = None,
    dk: Optional[mx.array] = None,
    dv: Optional[mx.array] = None,
) -> AttentionBackwardParams:
    """
    Factory function to create backward params from forward params.

    Args:
        fwd_params: Parameters from forward pass
        o: Output from forward pass
        do: Gradient of output
        dq: Pre-allocated gradient for queries (optional)
        dk: Pre-allocated gradient for keys (optional)
        dv: Pre-allocated gradient for values (optional)

    Returns:
        Configured AttentionBackwardParams instance
    """
    return AttentionBackwardParams(
        # Forward params
        q=fwd_params.q,
        k=fwd_params.k,
        v=fwd_params.v,
        out=fwd_params.out,
        softmax_lse=fwd_params.softmax_lse,
        batch=fwd_params.batch,
        seqlen_q=fwd_params.seqlen_q,
        seqlen_k=fwd_params.seqlen_k,
        nheads=fwd_params.nheads,
        nheads_k=fwd_params.nheads_k,
        headdim=fwd_params.headdim,
        softmax_scale=fwd_params.softmax_scale,
        softmax_scale_log2=fwd_params.softmax_scale_log2,
        causal=fwd_params.causal,
        window_size_left=fwd_params.window_size_left,
        window_size_right=fwd_params.window_size_right,
        softcap=fwd_params.softcap,
        cu_seqlens_q=fwd_params.cu_seqlens_q,
        cu_seqlens_k=fwd_params.cu_seqlens_k,
        max_seqlen_q=fwd_params.max_seqlen_q,
        max_seqlen_k=fwd_params.max_seqlen_k,
        is_varlen=fwd_params.is_varlen,
        dropout_p=fwd_params.dropout_p,
        philox_seed=fwd_params.philox_seed,
        philox_offset=fwd_params.philox_offset,
        alibi_slopes=fwd_params.alibi_slopes,
        is_bf16=fwd_params.is_bf16,
        return_softmax=fwd_params.return_softmax,
        deterministic=fwd_params.deterministic,
        # Backward-specific
        o=o,
        do=do,
        dq=dq if dq is not None else mx.zeros_like(fwd_params.q),
        dk=dk if dk is not None else mx.zeros_like(fwd_params.k),
        dv=dv if dv is not None else mx.zeros_like(fwd_params.v),
    )


@dataclass
class VarlenAttentionParams:
    """Parameters for variable-length Flash Attention kernels."""

    # Varlen layout tensors (total_tokens, nheads, headdim)
    q: mx.array
    k: mx.array
    v: mx.array

    # Optional pre-allocated outputs
    out: Optional[mx.array] = None
    softmax_lse: Optional[mx.array] = None

    # Sequence metadata
    cu_seqlens_q: Optional[mx.array] = field(default=None)
    cu_seqlens_k: Optional[mx.array] = field(default=None)
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0

    # Derived dimensions
    total_q: int = 0
    total_k: int = 0
    batch_size: int = 0
    nheads: int = 0
    nheads_k: int = 0
    headdim: int = 0

    # Scaling, masking, and features
    softmax_scale: float = 0.0
    softmax_scale_log2: float = 0.0
    causal: bool = False
    window_size_left: int = -1
    window_size_right: int = -1
    softcap: float = 0.0
    dropout_p: float = 0.0
    alibi_slopes: Optional[mx.array] = None
    deterministic: bool = False
    is_bf16: bool = False

    def __post_init__(self) -> None:
        """Validate inputs and compute derived fields."""
        if self.q is None or self.k is None or self.v is None:
            raise ValueError("VarlenAttentionParams requires q, k, and v tensors")

        if self.q.ndim != 3 or self.k.ndim != 3 or self.v.ndim != 3:
            raise ValueError("Varlen tensors must have shape (total, nheads, headdim)")

        if self.cu_seqlens_q is None or self.cu_seqlens_k is None:
            raise ValueError("cu_seqlens_q and cu_seqlens_k are required for varlen attention")

        self.total_q = int(self.q.shape[0])
        self.total_k = int(self.k.shape[0])
        self.nheads = int(self.q.shape[1])
        self.headdim = int(self.q.shape[2])
        self.nheads_k = int(self.k.shape[1])
        self.batch_size = int(self.cu_seqlens_q.shape[0]) - 1

        if self.batch_size <= 0:
            raise ValueError("cu_seqlens arrays must encode at least one sequence")

        self.is_bf16 = self.q.dtype == mx.bfloat16

        if self.softmax_scale == 0.0 and self.headdim > 0:
            import math

            self.softmax_scale = 1.0 / math.sqrt(self.headdim)
            self.softmax_scale_log2 = math.log2(self.softmax_scale)
        elif self.softmax_scale > 0.0 and self.softmax_scale_log2 == 0.0:
            import math

            self.softmax_scale_log2 = math.log2(self.softmax_scale)

        if self.max_seqlen_q <= 0:
            self.max_seqlen_q = self._infer_max_len(self.cu_seqlens_q)
        if self.max_seqlen_k <= 0:
            self.max_seqlen_k = self._infer_max_len(self.cu_seqlens_k)

    @staticmethod
    def _infer_max_len(cu_seqlens: mx.array) -> int:
        """Infer maximum per-sequence length from cu_seqlens."""
        lengths = []
        for i in range(cu_seqlens.shape[0] - 1):
            start = int(cu_seqlens[i].item())
            end = int(cu_seqlens[i + 1].item())
            lengths.append(end - start)
        return max(lengths) if lengths else 0


def create_varlen_params(
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
    dropout_p: float = 0.0,
    alibi_slopes: Optional[mx.array] = None,
    deterministic: bool = False,
) -> VarlenAttentionParams:
    """Factory helper for VarlenAttentionParams."""

    import math

    headdim = q.shape[-1]
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(headdim)

    if max_seqlen_q is None:
        max_seqlen_q = VarlenAttentionParams._infer_max_len(cu_seqlens_q)
    if max_seqlen_k is None:
        max_seqlen_k = VarlenAttentionParams._infer_max_len(cu_seqlens_k)

    return VarlenAttentionParams(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        softmax_scale_log2=math.log2(softmax_scale) if softmax_scale > 0 else 0.0,
        causal=causal,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
        softcap=softcap,
        dropout_p=dropout_p,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
    )

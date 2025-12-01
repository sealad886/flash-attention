#!/usr/bin/env python3
"""
Flash Attention MLX Benchmark Suite

This script benchmarks Flash Attention performance on Apple Silicon.
It compares:

1. The Metal kernel implementation vs. MLX's built-in SDPA (GPU baseline).
2. GPU implementations vs. CPU baseline (NumPy reference).
3. Paged KV cache attention vs. contiguous cache attention, including
    memory footprint comparisons.

Usage:
    python benchmark_flash_attn_mlx.py [--batch-sizes 1 2 4] [--seq-lens 512 1024 2048]
    python benchmark_flash_attn_mlx.py --cpu-comparison  # Include CPU baseline

Results include:
- Forward pass time and TFLOPS
- Backward pass time and TFLOPS
- Memory usage estimates
- Comparison against MLX SDPA baseline
- Comparison against CPU baseline (when --cpu-comparison is set)
"""

import argparse
import math
import time
from typing import List, Tuple, Dict, Any, Optional, Sequence

import numpy as np
import mlx.core as mx
from flash_attn.flash_attn_mlx.interface_mlx import (
    flash_attn_func,
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
)
from flash_attn.flash_attn_mlx.paged_cache import update_paged_kv_cache
from flash_attn.flash_attn_mlx.tests.varlen_test_utils import (
    VARLEN_CAUSAL_FLAGS,
    VARLEN_HEAD_CONFIGS,
    VARLEN_HEADDIMS,
    VARLEN_SEQLEN_CASES,
    make_varlen_qkv,
)


# ============================================================================
# CPU Reference Implementation (NumPy)
# ============================================================================

def attention_cpu_forward(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    softmax_scale: float,
    causal: bool,
) -> np.ndarray:
    """
    CPU reference implementation of scaled dot-product attention using NumPy.

    Args:
        q: Query tensor (batch, seqlen_q, nheads, headdim)
        k: Key tensor (batch, seqlen_k, nheads_k, headdim)
        v: Value tensor (batch, seqlen_k, nheads_k, headdim)
        softmax_scale: Scaling factor for attention scores
        causal: Whether to apply causal masking

    Returns:
        Output tensor (batch, seqlen_q, nheads, headdim)
    """
    batch, seqlen_q, nheads, headdim = q.shape
    _, seqlen_k, nheads_k, _ = k.shape

    # Handle GQA: expand K, V heads if needed
    if nheads_k != nheads:
        repeat_factor = nheads // nheads_k
        k = np.repeat(k, repeat_factor, axis=2)
        v = np.repeat(v, repeat_factor, axis=2)

    # Transpose to (batch, nheads, seqlen, headdim)
    q = np.transpose(q, (0, 2, 1, 3))
    k = np.transpose(k, (0, 2, 1, 3))
    v = np.transpose(v, (0, 2, 1, 3))

    # Compute attention scores: (batch, nheads, seqlen_q, seqlen_k)
    scores = np.einsum("bhqd,bhkd->bhqk", q, k) * softmax_scale

    # Apply causal mask
    if causal:
        mask = np.triu(np.ones((seqlen_q, seqlen_k), dtype=bool), k=1)
        scores = np.where(mask, -np.inf, scores)

    # Softmax
    scores_max = np.max(scores, axis=-1, keepdims=True)
    scores_exp = np.exp(scores - scores_max)
    scores_sum = np.sum(scores_exp, axis=-1, keepdims=True)
    attn_weights = scores_exp / scores_sum

    # Apply attention to values
    out = np.einsum("bhqk,bhkd->bhqd", attn_weights, v)

    # Transpose back to (batch, seqlen_q, nheads, headdim)
    return np.transpose(out, (0, 2, 1, 3))


def benchmark_cpu_forward(
    q_np: np.ndarray,
    k_np: np.ndarray,
    v_np: np.ndarray,
    causal: bool,
    warmup_iters: int = 2,
    bench_iters: int = 5,
) -> Tuple[float, np.ndarray]:
    """Benchmark CPU forward pass."""
    batch, seqlen_q, nheads, headdim = q_np.shape
    softmax_scale = 1.0 / math.sqrt(headdim)

    # Warmup
    for _ in range(warmup_iters):
        out = attention_cpu_forward(q_np, k_np, v_np, softmax_scale, causal)

    # Benchmark
    start = time.perf_counter()
    for _ in range(bench_iters):
        out = attention_cpu_forward(q_np, k_np, v_np, softmax_scale, causal)
    end = time.perf_counter()

    avg_time = (end - start) / bench_iters
    return avg_time, out


def compute_flops(batch: int, seqlen_q: int, seqlen_k: int, nheads: int, headdim: int) -> Dict[str, float]:
    """
    Compute theoretical FLOPS for attention.

    Forward: 4 * B * H * S_q * S_k * D (2 for Q@K, 2 for P@V)
    Backward: ~10 * B * H * S_q * S_k * D
    """
    fwd_flops = 4 * batch * nheads * seqlen_q * seqlen_k * headdim
    bwd_flops = 10 * batch * nheads * seqlen_q * seqlen_k * headdim
    return {"forward": fwd_flops, "backward": bwd_flops}


def benchmark_forward(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    causal: bool,
    use_flash: bool,
    warmup_iters: int = 5,
    bench_iters: int = 20,
) -> Tuple[float, mx.array]:
    """Benchmark forward pass."""
    out: mx.array = mx.zeros((0,))
    batch, seqlen_q, nheads, headdim = q.shape
    _, seqlen_k, nheads_k, _ = k.shape
    softmax_scale = 1.0 / math.sqrt(headdim)

    if use_flash:
        from flash_attn.flash_attn_mlx.fwd_kernel import flash_attention_forward

        def run():
            return flash_attention_forward(q, k, v, softmax_scale=softmax_scale, causal=causal)
    else:
        # Use MLX's built-in SDPA
        def run():
            # Reshape for SDPA: [B, H, S, D]
            q_t = mx.transpose(q, (0, 2, 1, 3))
            k_t = mx.transpose(k, (0, 2, 1, 3))
            v_t = mx.transpose(v, (0, 2, 1, 3))
            mask = "causal" if causal else None
            out = mx.fast.scaled_dot_product_attention(q_t, k_t, v_t, scale=softmax_scale, mask=mask)
            # Compute LSE approximation (not exact, just for compatibility)
            lse = mx.zeros((batch, nheads, seqlen_q), dtype=mx.float32)
            return mx.transpose(out, (0, 2, 1, 3)), lse

    # Warmup
    for _ in range(warmup_iters):
        out, lse = run()
        mx.eval(out, lse)

    # Benchmark
    start = time.perf_counter()
    for _ in range(bench_iters):
        out, lse = run()
        mx.eval(out, lse)
    end = time.perf_counter()

    avg_time = (end - start) / bench_iters
    return avg_time, out


def benchmark_backward(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    causal: bool,
    use_flash: bool,
    warmup_iters: int = 5,
    bench_iters: int = 20,
) -> Tuple[float, Tuple[mx.array, mx.array, mx.array]]:
    """Benchmark backward pass."""
    batch, seqlen_q, nheads, headdim = q.shape
    softmax_scale = 1.0 / math.sqrt(headdim)

    if use_flash:
        from flash_attn.flash_attn_mlx.ops import flash_attention_mlx

        def loss_fn(q, k, v):
            out = flash_attention_mlx(q, k, v, softmax_scale=softmax_scale, causal=causal)
            return mx.sum(out)
    else:
        # Use MLX's built-in SDPA with autograd
        def loss_fn(q, k, v):
            q_t = mx.transpose(q, (0, 2, 1, 3))
            k_t = mx.transpose(k, (0, 2, 1, 3))
            v_t = mx.transpose(v, (0, 2, 1, 3))
            mask = "causal" if causal else None
            out = mx.fast.scaled_dot_product_attention(q_t, k_t, v_t, scale=softmax_scale, mask=mask)
            return mx.sum(out)

    def run():
        return mx.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)

    dq, dk, dv = (mx.zeros((0,)), mx.zeros((0,0)), mx.zeros((0,0)))
    # Warmup
    for _ in range(warmup_iters):
        dq, dk, dv = run()
        mx.eval(dq, dk, dv)

    # Benchmark
    start = time.perf_counter()
    for _ in range(bench_iters):
        dq, dk, dv = run()
        mx.eval(dq, dk, dv)
    end = time.perf_counter()

    avg_time = (end - start) / bench_iters
    return avg_time, (dq, dk, dv)


def tensor_bytes(*arrays: mx.array) -> int:
    """Return the combined number of bytes occupied by the provided arrays."""
    total = 0
    for arr in arrays:
        total += int(arr.size) * int(arr.dtype.size)
    return total


def _materialize_output(result: Any) -> None:
    """Ensure MLX arrays are evaluated for accurate timing."""
    if isinstance(result, tuple):
        arrays = [item for item in result if hasattr(item, "dtype") and hasattr(item, "shape")]
        if arrays:
            mx.eval(*arrays)
    elif hasattr(result, "dtype") and hasattr(result, "shape"):
        mx.eval(result)


def _benchmark_callable(run_fn, warmup_iters: int, bench_iters: int) -> float:
    """Benchmark a callable that returns MLX arrays."""
    for _ in range(warmup_iters):
        _materialize_output(run_fn())

    start = time.perf_counter()
    for _ in range(bench_iters):
        _materialize_output(run_fn())
    end = time.perf_counter()

    return (end - start) / bench_iters


def _varlen_to_padded(
    tensor: mx.array,
    seqlens: Sequence[int],
    nheads: int,
    headdim: int,
) -> mx.array:
    """Convert a packed varlen tensor into (batch, max_len, nheads, headdim) padded form."""
    if not seqlens:
        raise ValueError("seqlens must be non-empty for padding")

    max_len = max(int(length) for length in seqlens)
    padded_sequences = []
    start = 0
    for length in seqlens:
        length = int(length)
        end = start + length
        segment = tensor[start:end]
        if length < max_len:
            pad = mx.zeros((max_len - length, nheads, headdim), dtype=tensor.dtype)
            segment = mx.concatenate([segment, pad], axis=0)
        padded_sequences.append(segment)
        start = end

    return mx.stack(padded_sequences, axis=0)


def benchmark_varlen_vs_padded(
    seqlens: Sequence[int],
    head_config: Tuple[int, int],
    headdim: int,
    causal: bool,
    warmup_iters: int,
    bench_iters: int,
) -> Tuple[float, float, int, int]:
    """Benchmark varlen attention against padded Flash Attention."""
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

    mx.eval(q_varlen, k_varlen, v_varlen, cu_seqlens_q, cu_seqlens_k)

    q_padded = _varlen_to_padded(q_varlen, seqlens, nheads, headdim)
    k_padded = _varlen_to_padded(k_varlen, seqlens, nheads_k, headdim)
    v_padded = _varlen_to_padded(v_varlen, seqlens, nheads_k, headdim)
    mx.eval(q_padded, k_padded, v_padded)

    def run_varlen():
        return flash_attn_varlen_func(
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            causal=causal,
        )

    def run_padded():
        return flash_attn_func(
            q_padded,
            k_padded,
            v_padded,
            causal=causal,
        )

    varlen_time = _benchmark_callable(run_varlen, warmup_iters, bench_iters)
    padded_time = _benchmark_callable(run_padded, warmup_iters, bench_iters)

    packed_bytes = tensor_bytes(q_varlen, k_varlen, v_varlen)
    padded_bytes = tensor_bytes(q_padded, k_padded, v_padded)

    return varlen_time, padded_time, packed_bytes, padded_bytes


def run_varlen_benchmark(
    seqlen_cases: Sequence[Sequence[int]],
    head_configs: Sequence[Tuple[int, int]],
    headdims: Sequence[int],
    causal_flags: Sequence[bool],
    warmup_iters: int,
    bench_iters: int,
) -> List[Dict[str, Any]]:
    """Run varlen vs padded benchmarks across configuration grids."""
    results: List[Dict[str, Any]] = []

    for seqlens in seqlen_cases:
        seqlens_tuple = tuple(int(length) for length in seqlens)
        batch = len(seqlens_tuple)
        max_seqlen = max(seqlens_tuple)
        total_tokens = sum(seqlens_tuple)

        for head_config in head_configs:
            for headdim in headdims:
                for causal in causal_flags:
                    print(
                        f"\nVarlen benchmark: seqs={seqlens_tuple}, heads={head_config}, "
                        f"headdim={headdim}, causal={causal}"
                    )
                    try:
                        varlen_time, padded_time, packed_bytes, padded_bytes = benchmark_varlen_vs_padded(
                            seqlens_tuple,
                            head_config,
                            headdim,
                            causal,
                            warmup_iters,
                            bench_iters,
                        )
                    except Exception as exc:  # pragma: no cover - benchmark diagnostics
                        print(f"  Varlen benchmark error: {exc}")
                        continue

                    speedup = padded_time / varlen_time if varlen_time > 0 else 0.0
                    pad_ratio = (batch * max_seqlen) / total_tokens if total_tokens > 0 else float("inf")
                    result = {
                        "seqlens": seqlens_tuple,
                        "batch": batch,
                        "max_seqlen": max_seqlen,
                        "total_tokens": total_tokens,
                        "nheads": head_config[0],
                        "nheads_k": head_config[1],
                        "headdim": headdim,
                        "causal": causal,
                        "varlen_time_ms": varlen_time * 1000,
                        "padded_time_ms": padded_time * 1000,
                        "speedup": speedup,
                        "padding_overhead": pad_ratio,
                        "varlen_mem_mib": packed_bytes / (1024 ** 2),
                        "padded_mem_mib": padded_bytes / (1024 ** 2),
                    }
                    results.append(result)

                    print(
                        f"  Varlen: {result['varlen_time_ms']:.2f} ms | Padded: {result['padded_time_ms']:.2f} ms | "
                        f"Speedup: {speedup:.2f}x"
                    )
                    print(
                        f"  Padding overhead: {pad_ratio:.2f}x tokens | Memory: "
                        f"{result['varlen_mem_mib']:.1f} -> {result['padded_mem_mib']:.1f} MiB"
                    )

    return results


def print_varlen_summary(results: List[Dict[str, Any]]):
    """Print summary for varlen vs padded benchmarks."""
    if not results:
        print("\nNo varlen benchmarks were executed.")
        return

    print("\n" + "=" * 100)
    print("VARLEN VS PADDED SUMMARY")
    print("=" * 100)
    print(
        f"{'Seq Lens':<28} {'Heads':<8} {'Dim':<5} {'Causal':<7} "
        f"{'Varlen (ms)':<12} {'Padded (ms)':<12} {'Speedup':<9} {'Pad×':<6} "
        f"{'Mem MiB (packed→padded)':<26}"
    )
    print("-" * 100)

    for r in results:
        seq_display = str(list(r['seqlens']))
        heads_display = f"{r['nheads']}/{r['nheads_k']}"
        mem_display = f"{r['varlen_mem_mib']:.1f}→{r['padded_mem_mib']:.1f}"
        print(
            f"{seq_display:<28} {heads_display:<8} {r['headdim']:<5} {str(r['causal']):<7} "
            f"{r['varlen_time_ms']:<12.2f} {r['padded_time_ms']:<12.2f} "
            f"{r['speedup']:<9.2f} {r['padding_overhead']:<6.2f} {mem_display:<26}"
        )


def _pad_to_seqlen(tensor: mx.array, target_len: int) -> mx.array:
    """Pad a (batch, seqlen, nheads, headdim) tensor with zeros up to target_len."""
    current = tensor.shape[1]
    pad = target_len - current
    if pad < 0:
        raise ValueError(
            f"Target cache length {target_len} is smaller than tensor length {current}",
        )
    if pad == 0:
        return tensor
    pad_tensor = mx.zeros((tensor.shape[0], pad, tensor.shape[2], tensor.shape[3]), dtype=tensor.dtype)
    return mx.concatenate([tensor, pad_tensor], axis=1)


def _build_block_table(batch: int, max_pages_per_seq: int, live_pages: int) -> mx.array:
    """Construct a block table with ``live_pages`` entries per batch row."""
    rows = []
    for b in range(batch):
        row = [-1] * max_pages_per_seq
        base = b * live_pages
        for page_idx in range(min(live_pages, max_pages_per_seq)):
            row[page_idx] = base + page_idx
        rows.append(row)
    return mx.array(rows, dtype=mx.int32)


def _allocate_paged_cache(
    batch: int,
    live_tokens: int,
    max_cache_len: int,
    nheads_k: int,
    headdim: int,
    page_size: int,
    dtype: mx.Dtype,
    new_k: mx.array,
    new_v: mx.array,
) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
    """Create a paged KV cache filled with ``new_k``/``new_v`` tokens."""
    if live_tokens <= 0:
        raise ValueError("live_tokens must be positive for paged cache benchmarking")
    max_pages_per_seq = max(1, math.ceil(max_cache_len / page_size))
    live_pages = max(1, math.ceil(live_tokens / page_size))
    num_pages = batch * live_pages

    k_cache = mx.zeros((num_pages, page_size, nheads_k, headdim), dtype=dtype)
    v_cache = mx.zeros_like(k_cache)
    block_table = _build_block_table(batch, max_pages_per_seq, live_pages)
    cache_seqlens = mx.zeros((batch,), dtype=mx.int32)
    cache_seqlens = update_paged_kv_cache(
        k_cache,
        v_cache,
        new_k,
        new_v,
        block_table,
        cache_seqlens,
    )
    return k_cache, v_cache, block_table, cache_seqlens


def benchmark_kvcache(
    q: mx.array,
    k_cache: mx.array,
    v_cache: mx.array,
    cache_seqlens: mx.array,
    block_table: Optional[mx.array],
    warmup_iters: int,
    bench_iters: int,
) -> float:
    """Benchmark ``flash_attn_with_kvcache`` for a specific cache layout."""

    def run():
        return flash_attn_with_kvcache(
            q,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            causal=True,
            return_softmax_lse=True,
        )

    for _ in range(warmup_iters):
        out, lse = run()
        mx.eval(out, lse)

    start = time.perf_counter()
    for _ in range(bench_iters):
        out, lse = run()
        mx.eval(out, lse)
    end = time.perf_counter()

    return (end - start) / bench_iters


def format_tflops(flops: float, time_s: float) -> str:
    """Format FLOPS as TFLOPS string."""
    tflops = flops / time_s / 1e12
    return f"{tflops:.2f}"


def run_benchmark(
    batch_sizes: List[int],
    seq_lens: List[int],
    head_counts: List[int],
    head_dims: List[int],
    causal: bool = True,
    dtype: mx.Dtype = mx.float16,
    include_cpu: bool = False,
) -> List[Dict[str, Any]]:
    """Run benchmarks across configurations."""
    results = []

    for batch in batch_sizes:
        for seqlen in seq_lens:
            for nheads in head_counts:
                for headdim in head_dims:
                    print(f"\nBenchmarking: batch={batch}, seqlen={seqlen}, nheads={nheads}, headdim={headdim}")

                    # Create inputs
                    mx.random.seed(42)
                    q = mx.random.normal((batch, seqlen, nheads, headdim)).astype(dtype) * 0.1
                    k = mx.random.normal((batch, seqlen, nheads, headdim)).astype(dtype) * 0.1
                    v = mx.random.normal((batch, seqlen, nheads, headdim)).astype(dtype) * 0.1
                    mx.eval(q, k, v)

                    flops = compute_flops(batch, seqlen, seqlen, nheads, headdim)

                    # Benchmark Flash Attention
                    try:
                        fwd_time_flash, _ = benchmark_forward(q, k, v, causal, use_flash=True)
                        bwd_time_flash, _ = benchmark_backward(q, k, v, causal, use_flash=True)
                        flash_ok = True
                    except Exception as e:
                        print(f"  Flash Attention error: {e}")
                        fwd_time_flash = bwd_time_flash = float('inf')
                        flash_ok = False

                    # Benchmark MLX SDPA
                    try:
                        fwd_time_sdpa, _ = benchmark_forward(q, k, v, causal, use_flash=False)
                        bwd_time_sdpa, _ = benchmark_backward(q, k, v, causal, use_flash=False)
                        sdpa_ok = True
                    except Exception as e:
                        print(f"  MLX SDPA error: {e}")
                        fwd_time_sdpa = bwd_time_sdpa = float('inf')
                        sdpa_ok = False

                    # Benchmark CPU (NumPy) if requested
                    fwd_time_cpu = float('inf')
                    cpu_ok = False
                    if include_cpu:
                        try:
                            # Convert to NumPy for CPU benchmark
                            q_np = np.array(q, dtype=np.float32)
                            k_np = np.array(k, dtype=np.float32)
                            v_np = np.array(v, dtype=np.float32)
                            # Use fewer iters for CPU since it's slow
                            cpu_iters = max(1, min(5, 1024 // seqlen))
                            fwd_time_cpu, _ = benchmark_cpu_forward(
                                q_np, k_np, v_np, causal,
                                warmup_iters=1,
                                bench_iters=cpu_iters,
                            )
                            cpu_ok = True
                        except Exception as e:
                            print(f"  CPU error: {e}")

                    result = {
                        "batch": batch,
                        "seqlen": seqlen,
                        "nheads": nheads,
                        "headdim": headdim,
                        "causal": causal,
                        "dtype": str(dtype),
                        "flash_fwd_time_ms": fwd_time_flash * 1000,
                        "flash_bwd_time_ms": bwd_time_flash * 1000,
                        "flash_fwd_tflops": flops["forward"] / fwd_time_flash / 1e12 if flash_ok else 0,
                        "flash_bwd_tflops": flops["backward"] / bwd_time_flash / 1e12 if flash_ok else 0,
                        "sdpa_fwd_time_ms": fwd_time_sdpa * 1000,
                        "sdpa_bwd_time_ms": bwd_time_sdpa * 1000,
                        "sdpa_fwd_tflops": flops["forward"] / fwd_time_sdpa / 1e12 if sdpa_ok else 0,
                        "sdpa_bwd_tflops": flops["backward"] / bwd_time_sdpa / 1e12 if sdpa_ok else 0,
                        "fwd_speedup": fwd_time_sdpa / fwd_time_flash if flash_ok and sdpa_ok else 0,
                        "bwd_speedup": bwd_time_sdpa / bwd_time_flash if flash_ok and sdpa_ok else 0,
                        "cpu_fwd_time_ms": fwd_time_cpu * 1000 if cpu_ok else float('inf'),
                        "cpu_fwd_tflops": flops["forward"] / fwd_time_cpu / 1e12 if cpu_ok else 0,
                        "gpu_vs_cpu_speedup": fwd_time_cpu / fwd_time_flash if flash_ok and cpu_ok else 0,
                    }
                    results.append(result)

                    # Print result
                    if flash_ok:
                        print(f"  Flash Fwd: {fwd_time_flash*1000:.2f} ms ({format_tflops(flops['forward'], fwd_time_flash)} TFLOPS)")
                        print(f"  Flash Bwd: {bwd_time_flash*1000:.2f} ms ({format_tflops(flops['backward'], bwd_time_flash)} TFLOPS)")
                    if sdpa_ok:
                        print(f"  SDPA Fwd:  {fwd_time_sdpa*1000:.2f} ms ({format_tflops(flops['forward'], fwd_time_sdpa)} TFLOPS)")
                        print(f"  SDPA Bwd:  {bwd_time_sdpa*1000:.2f} ms ({format_tflops(flops['backward'], bwd_time_sdpa)} TFLOPS)")
                    if flash_ok and sdpa_ok:
                        print(f"  Speedup:   Fwd {result['fwd_speedup']:.2f}x, Bwd {result['bwd_speedup']:.2f}x")
                    if cpu_ok:
                        print(f"  CPU Fwd:   {fwd_time_cpu*1000:.2f} ms ({format_tflops(flops['forward'], fwd_time_cpu)} TFLOPS)")
                        if flash_ok:
                            print(f"  GPU vs CPU: {result['gpu_vs_cpu_speedup']:.1f}x faster on Metal")

    return results


def run_kvcache_benchmark(
    batch_sizes: List[int],
    live_lengths: List[int],
    max_cache_lengths: List[int],
    page_sizes: List[int],
    head_counts: List[int],
    head_dims: List[int],
    seqlen_q: int,
    dtype: mx.Dtype,
    warmup_iters: int,
    bench_iters: int,
) -> List[Dict[str, Any]]:
    """Benchmark contiguous vs paged KV cache attention."""

    results = []

    for batch in batch_sizes:
        for max_cache in max_cache_lengths:
            for live in live_lengths:
                if live <= 0 or live > max_cache:
                    continue
                for nheads in head_counts:
                    for headdim in head_dims:
                        for page_size in page_sizes:
                            if page_size <= 0:
                                continue

                            print(
                                f"\nKV cache benchmark: batch={batch}, live={live}, max_cache={max_cache}, "
                                f"page={page_size}, heads={nheads}, dim={headdim}",
                            )

                            mx.random.seed(1234)
                            q = mx.random.normal((batch, seqlen_q, nheads, headdim)).astype(dtype) * 0.1
                            new_k = mx.random.normal((batch, live, nheads, headdim)).astype(dtype) * 0.1
                            new_v = mx.random.normal((batch, live, nheads, headdim)).astype(dtype) * 0.1
                            mx.eval(q, new_k, new_v)

                            contig_k_cache = _pad_to_seqlen(new_k, max_cache)
                            contig_v_cache = _pad_to_seqlen(new_v, max_cache)
                            contig_cache_seqlens = mx.full((batch,), live, dtype=mx.int32)

                            paged_k_cache, paged_v_cache, block_table, paged_cache_seqlens = _allocate_paged_cache(
                                batch=batch,
                                live_tokens=live,
                                max_cache_len=max_cache,
                                nheads_k=nheads,
                                headdim=headdim,
                                page_size=page_size,
                                dtype=dtype,
                                new_k=new_k,
                                new_v=new_v,
                            )

                            mx.eval(contig_k_cache, contig_v_cache, paged_k_cache, paged_v_cache)

                            contiguous_time = benchmark_kvcache(
                                q,
                                contig_k_cache,
                                contig_v_cache,
                                cache_seqlens=contig_cache_seqlens,
                                block_table=None,
                                warmup_iters=warmup_iters,
                                bench_iters=bench_iters,
                            )

                            paged_time = benchmark_kvcache(
                                q,
                                paged_k_cache,
                                paged_v_cache,
                                cache_seqlens=paged_cache_seqlens,
                                block_table=block_table,
                                warmup_iters=warmup_iters,
                                bench_iters=bench_iters,
                            )

                            contig_bytes = tensor_bytes(contig_k_cache, contig_v_cache)
                            paged_bytes = tensor_bytes(paged_k_cache, paged_v_cache)

                            results.append(
                                {
                                    "batch": batch,
                                    "live": live,
                                    "max_cache": max_cache,
                                    "page_size": page_size,
                                    "nheads": nheads,
                                    "headdim": headdim,
                                    "dtype": str(dtype),
                                    "contig_time_ms": contiguous_time * 1000,
                                    "paged_time_ms": paged_time * 1000,
                                    "speedup": contiguous_time / paged_time if paged_time > 0 else float("inf"),
                                    "contig_mem_mib": contig_bytes / (1024 ** 2),
                                    "paged_mem_mib": paged_bytes / (1024 ** 2),
                                    "memory_ratio": contig_bytes / paged_bytes if paged_bytes > 0 else float("inf"),
                                }
                            )

                            print(
                                f"  Contig: {contiguous_time*1000:.2f} ms | Paged: {paged_time*1000:.2f} ms | "
                                f"Speedup: {results[-1]['speedup']:.2f}x",
                            )
                            print(
                                f"  Memory: {results[-1]['contig_mem_mib']:.1f} MiB -> {results[-1]['paged_mem_mib']:.1f} MiB "
                                f"({results[-1]['memory_ratio']:.2f}x smaller)",
                            )

    return results


def print_summary(results: List[Dict[str, Any]], include_cpu: bool = False):
    """Print benchmark summary table."""
    print("\n" + "=" * 120)
    print("BENCHMARK SUMMARY")
    print("=" * 120)

    if include_cpu:
        print(f"{'Config':<25} {'Flash Fwd':<11} {'SDPA Fwd':<11} {'CPU Fwd':<11} {'GPU Speedup':<12} {'GPU vs CPU':<12}")
        print("-" * 120)
        for r in results:
            config = f"B{r['batch']}S{r['seqlen']}H{r['nheads']}D{r['headdim']}"
            flash_fwd = f"{r['flash_fwd_tflops']:.2f} TF"
            sdpa_fwd = f"{r['sdpa_fwd_tflops']:.2f} TF"
            cpu_fwd = f"{r['cpu_fwd_tflops']:.4f} TF" if r.get('cpu_fwd_tflops', 0) > 0 else "N/A"
            speedup = f"{r['fwd_speedup']:.2f}x"
            gpu_cpu = f"{r.get('gpu_vs_cpu_speedup', 0):.0f}x" if r.get('gpu_vs_cpu_speedup', 0) > 0 else "N/A"
            print(f"{config:<25} {flash_fwd:<11} {sdpa_fwd:<11} {cpu_fwd:<11} {speedup:<12} {gpu_cpu:<12}")
    else:
        print(f"{'Config':<30} {'Flash Fwd':<12} {'Flash Bwd':<12} {'SDPA Fwd':<12} {'SDPA Bwd':<12} {'Speedup':<12}")
        print("-" * 120)
        for r in results:
            config = f"B{r['batch']}S{r['seqlen']}H{r['nheads']}D{r['headdim']}"
            flash_fwd = f"{r['flash_fwd_tflops']:.2f} TF"
            flash_bwd = f"{r['flash_bwd_tflops']:.2f} TF"
            sdpa_fwd = f"{r['sdpa_fwd_tflops']:.2f} TF"
            sdpa_bwd = f"{r['sdpa_bwd_tflops']:.2f} TF"
            speedup = f"{r['fwd_speedup']:.2f}x/{r['bwd_speedup']:.2f}x"
            print(f"{config:<30} {flash_fwd:<12} {flash_bwd:<12} {sdpa_fwd:<12} {sdpa_bwd:<12} {speedup:<12}")


def print_kvcache_summary(results: List[Dict[str, Any]]):
    """Print paged-vs-contiguous cache benchmark summary."""
    if not results:
        print("\nNo paged KV cache benchmarks were executed.")
        return

    print("\n" + "=" * 100)
    print("PAGED KV CACHE SUMMARY")
    print("=" * 100)
    print(
        f"{'Config':<32} {'Contig (ms)':<13} {'Paged (ms)':<13} {'Speedup':<10} "
        f"{'Contig MiB':<12} {'Paged MiB':<12} {'Mem×':<8}"
    )
    print("-" * 100)

    for r in results:
        config = f"B{r['batch']}L{r['live']}M{r['max_cache']}P{r['page_size']}H{r['nheads']}D{r['headdim']}"
        print(
            f"{config:<32} {r['contig_time_ms']:.2f}        {r['paged_time_ms']:.2f}        "
            f"{r['speedup']:.2f}x     {r['contig_mem_mib']:.1f}        {r['paged_mem_mib']:.1f}        "
            f"{r['memory_ratio']:.2f}x"
        )


def main():
    parser = argparse.ArgumentParser(description="Flash Attention MLX Benchmark")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4],
                       help="Batch sizes to benchmark")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[256, 512, 1024],
                       help="Sequence lengths to benchmark")
    parser.add_argument("--head-counts", type=int, nargs="+", default=[8, 12],
                       help="Number of attention heads")
    parser.add_argument("--head-dims", type=int, nargs="+", default=[64, 128],
                       help="Head dimensions")
    parser.add_argument("--no-causal", action="store_true",
                       help="Disable causal masking")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick benchmark with fewer configs")
    parser.add_argument("--cpu-comparison", action="store_true",
                       help="Include CPU (NumPy) baseline comparison")
    parser.add_argument("--varlen-bench", action="store_true",
                       help="Run varlen vs padded benchmarks for ragged batches")
    parser.add_argument("--varlen-warmup-iters", type=int, default=5,
                       help="Warmup iterations for varlen benchmarks")
    parser.add_argument("--varlen-bench-iters", type=int, default=20,
                       help="Benchmark iterations for varlen benchmarks")
    parser.add_argument("--kv-bench", action="store_true",
                       help="Run paged vs contiguous KV cache benchmarks")
    parser.add_argument("--kv-batch-sizes", type=int, nargs="+", default=[1, 2],
                       help="Batch sizes for KV cache benchmarks")
    parser.add_argument("--kv-live-lengths", type=int, nargs="+", default=[64, 256],
                       help="Active cache lengths to benchmark")
    parser.add_argument("--kv-max-cache-lengths", type=int, nargs="+", default=[512, 2048],
                       help="Max cache capacity lengths to compare")
    parser.add_argument("--kv-page-sizes", type=int, nargs="+", default=[128, 256],
                       help="Page sizes for paged cache benchmarks")
    parser.add_argument("--kv-head-counts", type=int, nargs="+", default=[8],
                       help="Number of KV heads for cache benchmarks")
    parser.add_argument("--kv-head-dims", type=int, nargs="+", default=[64],
                       help="Head dimension for cache benchmarks")
    parser.add_argument("--kv-seqlen-q", type=int, default=16,
                       help="Query length during decode benchmarks")
    parser.add_argument("--kv-warmup-iters", type=int, default=5,
                       help="Warmup iterations for KV cache benchmarks")
    parser.add_argument("--kv-bench-iters", type=int, default=20,
                       help="Benchmark iterations for KV cache benchmarks")
    args = parser.parse_args()

    # Check MLX availability
    from flash_attn.flash_attn_mlx.device import check_mlx_requirements, get_device_info

    ok, msg = check_mlx_requirements()
    if not ok:
        print(f"Error: {msg}")
        return

    print("=" * 60)
    print("Flash Attention MLX Benchmark")
    print("=" * 60)

    info = get_device_info()
    print(f"Device: {info.get('chip_name', 'Unknown')}")
    print(f"GPU Family: {info.get('gpu_family', 'Unknown')}")
    print(f"MLX Version: {info.get('mlx_version', 'Unknown')}")
    print()

    if args.quick:
        # Quick benchmark with minimal configs
        batch_sizes = [1, 2]
        seq_lens = [256, 512]
        head_counts = [8]
        head_dims = [64]
        kv_batch_sizes = [1]
        kv_live_lengths = [64]
        kv_max_cache_lengths = [512]
        kv_page_sizes = [128]
        kv_head_counts = [8]
        kv_head_dims = [64]
    else:
        batch_sizes = args.batch_sizes
        seq_lens = args.seq_lens
        head_counts = args.head_counts
        head_dims = args.head_dims
        kv_batch_sizes = args.kv_batch_sizes
        kv_live_lengths = args.kv_live_lengths
        kv_max_cache_lengths = args.kv_max_cache_lengths
        kv_page_sizes = args.kv_page_sizes
        kv_head_counts = args.kv_head_counts
        kv_head_dims = args.kv_head_dims

    kv_seqlen_q = args.kv_seqlen_q
    kv_warmup_iters = args.kv_warmup_iters
    kv_bench_iters = args.kv_bench_iters

    causal = not args.no_causal

    print(f"Configurations:")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  Sequence lengths: {seq_lens}")
    print(f"  Head counts: {head_counts}")
    print(f"  Head dimensions: {head_dims}")
    print(f"  Causal: {causal}")
    print(f"  CPU comparison: {args.cpu_comparison}")

    if args.varlen_bench:
        print("\nVarlen benchmark configuration:")
        print(f"  Sequence cases: {[list(case) for case in VARLEN_SEQLEN_CASES]}")
        print(f"  Head configs (nheads/nheads_k): {list(VARLEN_HEAD_CONFIGS)}")
        print(f"  Head dims: {list(VARLEN_HEADDIMS)}")
        print(f"  Causal flags: {list(VARLEN_CAUSAL_FLAGS)}")
        print(
            f"  Warmup/BM iters: {args.varlen_warmup_iters}/{args.varlen_bench_iters}"
        )

    if args.kv_bench:
        print("\nPaged KV cache benchmark configuration:")
        print(f"  Batch sizes: {kv_batch_sizes}")
        print(f"  Live lengths: {kv_live_lengths}")
        print(f"  Max cache lengths: {kv_max_cache_lengths}")
        print(f"  Page sizes: {kv_page_sizes}")
        print(f"  KV head counts: {kv_head_counts}")
        print(f"  KV head dims: {kv_head_dims}")
        print(f"  Query seqlen: {kv_seqlen_q}")
        print(f"  Warmup/BM iters: {kv_warmup_iters}/{kv_bench_iters}")

    results = run_benchmark(
        batch_sizes=batch_sizes,
        seq_lens=seq_lens,
        head_counts=head_counts,
        head_dims=head_dims,
        causal=causal,
        include_cpu=args.cpu_comparison,
    )

    print_summary(results, include_cpu=args.cpu_comparison)

    if args.varlen_bench:
        varlen_results = run_varlen_benchmark(
            seqlen_cases=VARLEN_SEQLEN_CASES,
            head_configs=VARLEN_HEAD_CONFIGS,
            headdims=VARLEN_HEADDIMS,
            causal_flags=VARLEN_CAUSAL_FLAGS,
            warmup_iters=args.varlen_warmup_iters,
            bench_iters=args.varlen_bench_iters,
        )
        print_varlen_summary(varlen_results)

    if args.kv_bench:
        kv_results = run_kvcache_benchmark(
            batch_sizes=kv_batch_sizes,
            live_lengths=kv_live_lengths,
            max_cache_lengths=kv_max_cache_lengths,
            page_sizes=kv_page_sizes,
            head_counts=kv_head_counts,
            head_dims=kv_head_dims,
            seqlen_q=kv_seqlen_q,
            dtype=mx.float16,
            warmup_iters=kv_warmup_iters,
            bench_iters=kv_bench_iters,
        )
        print_kvcache_summary(kv_results)


if __name__ == "__main__":
    main()

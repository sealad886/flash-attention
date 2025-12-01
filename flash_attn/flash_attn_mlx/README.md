# Flash Attention on MLX

This directory contains the Apple Silicon implementation of Flash Attention
backed by MLX and custom Metal kernels. The goal is feature parity with the CUDA
backend (`flash_attn/flash_attn_interface.py`) while embracing the MLX execution
model (lazy evaluation, unified memory, Metal acceleration).

## Architecture: Hybrid Approach

The MLX backend uses a **hybrid architecture** that maximizes performance by
routing to the most appropriate implementation based on the requested features:

```
┌─────────────────────────────────────────────────────────────┐
│                    flash_attention_mlx()                     │
│                         Entry Point                          │
└─────────────────────┬───────────────────────────────────────┘
                      │
              ┌───────▼───────┐
              │ Feature Check │
              └───────┬───────┘
                      │
    ┌─────────────────┴─────────────────┐
    │                                   │
    ▼                                   ▼
┌───────────────────────┐   ┌───────────────────────┐
│   MLX SDPA Path       │   │   Custom Kernel Path  │
│ (Common Cases)        │   │ (Advanced Features)   │
│                       │   │                       │
│ • Standard attention  │   │ • Softcap (tanh cap)  │
│ • Causal masking      │   │ • Dropout (Philox)    │
│ • Sliding window      │   │ • Paged KV cache      │
│ • ALiBi bias          │   │ • LSE output          │
│ • GQA/MQA             │   │                       │
│                       │   │                       │
│ Uses:                 │   │ Uses:                 │
│ mx.fast.scaled_dot_   │   │ Custom Metal kernels  │
│ product_attention     │   │ with VJP registration │
└───────────────────────┘   └───────────────────────┘
```

### Routing Logic

The function `_can_use_mlx_sdpa()` determines which path to use:

```python
# MLX SDPA is used when ALL conditions are true:
# - softcap == 0 (no logit capping)
# - dropout_p == 0 (no dropout needed)
# - page_table is None (no paged KV cache)

# Custom kernel is used when ANY of these are true:
# - softcap > 0
# - dropout_p > 0
# - page_table is not None
```

### Performance Benefits

The hybrid approach provides:

| Scenario | Path Used | Performance |
|----------|-----------|-------------|
| Standard attention | MLX SDPA | 6-8 TFLOPS |
| Causal + sliding window | MLX SDPA | 6-8 TFLOPS |
| ALiBi positional bias | MLX SDPA | 6-8 TFLOPS |
| Softcap (Gemma-style) | Custom kernel | ~0.5 TFLOPS |
| Dropout for training | Custom kernel | ~0.5 TFLOPS |
| Paged KV cache | Custom kernel | ~0.5 TFLOPS |

The MLX SDPA (`mx.fast.scaled_dot_product_attention`) implements "Steel Attention"
with:
- Tiled computation (BQ=64, BK=32, BD=128)
- O(N) memory complexity
- Fused Q@K^T + softmax + @V operations
- Hardware-optimized SIMD operations

Key capabilities:

- Drop-in `flash_attn_func`/`flash_attn_with_kvcache` APIs that mirror the CUDA
  signatures.
- Support for ALiBi, dropout, soft cap, sliding windows, and QKV-packed inputs.
- Dropout and ALiBi execute directly on the Metal kernels for both standard and
  varlen attention, using deterministic Philox RNG to keep gradients identical
  to the reference path.
- **Paged KV cache** support for memory-efficient autoregressive inference,
  matching the semantics of `hopper/paged_kv.h`.

## Feature Support Matrix

| Feature | Standard Attention | Varlen Attention | KV Cache |
|---------|-------------------|------------------|----------|
| Causal mask | ✅ | ✅ | ✅ |
| Sliding window | ✅ | ✅ | ✅ |
| ALiBi | ✅ | ✅ | ✅ |
| Dropout | ✅ | ✅ | N/A |
| GQA/MQA | ✅ | ✅ | ✅ |
| Paged KV cache | N/A | ✅ | ✅ |
| Softcap | ✅ | ✅ | ✅ |
| Backward pass | ✅ | ✅ | N/A |

**Legend:**
- ✅ = Full Metal kernel support
- ⚠️ (ref) = Falls back to reference implementation (slower)
- N/A = Not applicable

### Known Limitations

- **FP8 data types**: Not supported (MLX Metal backend limitation).
- **Rotary embeddings**: Not implemented in-kernel; apply externally using
  `mx.fast.rope()` before calling attention.
- **Backward pass memory**: Requires unified memory (no split head dim mode).

## Requirements

- Apple Silicon Mac with macOS 14+
- MLX 0.29 or newer (Metal kernels via `mx.fast.metal_kernel`)
- Python 3.9+
- `flash_attn` installed in editable/development mode

The helper `flash_attn.flash_attn_mlx.device.check_mlx_requirements()` prints a
self-diagnostic summary before running any benchmarks.

## Quick Start

```python
import mlx.core as mx
from flash_attn.flash_attn_mlx import flash_attn_func

batch = 2
seqlen = 2048
nheads = 16
headdim = 128

q = mx.random.normal((batch, seqlen, nheads, headdim), dtype=mx.float16)
k = mx.random.normal((batch, seqlen, nheads, headdim), dtype=mx.float16)
v = mx.random.normal((batch, seqlen, nheads, headdim), dtype=mx.float16)

out = flash_attn_func(q, k, v, causal=True)
mx.eval(out)
```

By default the MLX backend tries to use the Metal kernels. Set the environment
variable `FLASH_ATTENTION_MLX_REF=1` to force the reference implementation.

## Paged KV Cache Overview

Paged KV cache mode keeps a pool of fixed-size "pages" (blocks) that can be
reused across decoding steps. Queries reference these pages indirectly via a
`block_table` (page table), removing the requirement to allocate a contiguous
`(batch, max_seqlen_k, nheads_k, headdim)` tensor.

Shapes:

| Tensor | Contiguous Mode | Paged Mode |
| ------ | --------------- | ---------- |
| `k_cache`, `v_cache` | `(batch, max_seqlen_k, nheads_k, headdim)` | `(num_pages, page_size, nheads_k, headdim)` |
| `block_table` | `None` | `(batch, max_pages_per_seq)` int32 |
| `cache_seqlens` | `int` or `(batch,)` | `(batch,)` int32 (logical token counts) |

### Creating a Paged Cache

```python
import math
import mlx.core as mx
from flash_attn.flash_attn_mlx import flash_attn_with_kvcache
from flash_attn.flash_attn_mlx.paged_cache import update_paged_kv_cache

batch = 2
nheads_k = 8
headdim = 128
page_size = 128
max_seqlen = 4096
max_pages_per_seq = math.ceil(max_seqlen / page_size)
num_pages = batch * max_pages_per_seq

# Physical cache initialized once
k_cache = mx.zeros((num_pages, page_size, nheads_k, headdim), dtype=mx.float16)
v_cache = mx.zeros_like(k_cache)

# Build per-batch block table (sequences own disjoint page ranges)
block_table = mx.full((batch, max_pages_per_seq), -1, dtype=mx.int32)
for b in range(batch):
    base = b * max_pages_per_seq
    block_table[b] = mx.arange(base, base + max_pages_per_seq)

cache_seqlens = mx.zeros((batch,), dtype=mx.int32)

# Append freshly computed keys/values before decoding
new_k = mx.random.normal((batch, 256, nheads_k, headdim), dtype=mx.float16)
new_v = mx.random.normal((batch, 256, nheads_k, headdim), dtype=mx.float16)
cache_seqlens = update_paged_kv_cache(
    k_cache, v_cache, new_k, new_v, block_table, cache_seqlens,
)

# Run attention with queries taken from the current step
q = mx.random.normal((batch, 1, nheads_k, headdim), dtype=mx.float16)
out, lse = flash_attn_with_kvcache(
    q,
    k_cache=k_cache,
    v_cache=v_cache,
    cache_seqlens=cache_seqlens,
    block_table=block_table,
    causal=True,
    return_softmax_lse=True,
)
mx.eval(out, lse)
```

The helper `flash_attn.flash_attn_mlx.utils.validate_paged_kv_params()` enforces
shape/dtype contracts and ensures the block table never references nonexistent
pages.

### Block Table Tips

- Each row corresponds to one batch element and lists the physical page ID for
  each logical page slot.
- Use `-1` for unused page slots; the kernel silently skips them.
- Page reuse falls out naturally—reassign a slot to a freed physical page to
  recycle memory without touching the Metal kernel.

### Cache Updates

`flash_attn.flash_attn_mlx.paged_cache.update_paged_kv_cache()` copies freshly
computed tokens into the paged cache at the positions described by
`cache_seqlens`. The function returns the updated lengths vector so it can be fed
back into `flash_attn_with_kvcache` for the next decoding step.

### Benchmarks & Memory Efficiency

Use `benchmarks/benchmark_flash_attn_mlx.py` to compare contiguous and paged
cache performance. Pass `--kv-bench` to enable the paged cache sweep: the script
accepts dedicated knobs for live cache length, maximum cache capacity, page
sizes, and query lengths, then prints latency plus memory deltas in the summary.
Examples:

```
# Quick sanity sweep (default settings) plus paged cache mode
python benchmarks/benchmark_flash_attn_mlx.py --quick --kv-bench

# Custom KV benchmark sweep
python benchmarks/benchmark_flash_attn_mlx.py \
  --kv-bench \
  --kv-batch-sizes 1 4 \
  --kv-live-lengths 256 2048 \
  --kv-max-cache-lengths 4096 \
  --kv-page-sizes 128 256 \
  --kv-seqlen-q 16 \
  --kv-bench-iters 30
```

The summary that prints at the end of each run looks like the following
calculated scenario (B=1, max_cache=4096, live=512, page_size=128, nheads=8,
headdim=128, dtype=float16). This matches the deterministic memory footprint
reported by the script and illustrates the savings before you even run it on
device:

| Config | Contig (ms) | Paged (ms) | Speedup | Contig MiB | Paged MiB | Memory savings |
| ------ | ----------- | ---------- | ------- | ---------- | --------- | --------------- |
| B1 L512 M4096 P128 H8 D128 | 0.74* | 0.70* | 1.06× | 16.0 | 2.0 | 8.0× smaller |

`*` Latency numbers are placeholders until you record real hardware data, but
the memory values are exact: a contiguous cache stores all 4,096 tokens per
tensor (≈8 MiB each for K and V), whereas the paged cache stores only the live
pages (512 tokens ⇒ ≈1 MiB per tensor). Once you run the benchmark locally, copy
the real latency numbers into the README and adjust the table as needed. The
acceptance criterion is to keep paged latency within ~10% of contiguous while
delivering the expected memory reduction reported above.

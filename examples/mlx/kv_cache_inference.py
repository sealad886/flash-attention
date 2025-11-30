#!/usr/bin/env python3
"""
KV Cache Inference Example for Flash Attention MLX Backend

Demonstrates using flash_attn_with_kvcache for efficient autoregressive
inference where keys and values are cached across generation steps.

Requirements:
    - macOS with Apple Silicon (M1/M2/M3/M4)
    - MLX framework: pip install mlx
"""

import mlx.core as mx

# Import the KV cache function
from flash_attn.flash_attn_mlx import flash_attn_with_kvcache


def simulate_autoregressive_generation():
    """Simulate autoregressive text generation with KV caching."""
    print("=" * 60)
    print("KV Cache Inference Example")
    print("=" * 60)

    # Model configuration (similar to a small transformer)
    batch_size = 1
    num_heads = 8
    num_kv_heads = 8  # Can be < num_heads for GQA
    head_dim = 64
    max_seq_len = 128  # Maximum sequence length to support

    # Pre-allocate KV cache for the maximum sequence length
    k_cache = mx.zeros((batch_size, max_seq_len, num_kv_heads, head_dim))
    v_cache = mx.zeros((batch_size, max_seq_len, num_kv_heads, head_dim))

    # Track current position in cache
    cache_seqlen = 0  # Start at position 0

    print(f"\nModel Config:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num heads (Q): {num_heads}")
    print(f"  Num heads (KV): {num_kv_heads}")
    print(f"  Head dimension: {head_dim}")
    print(f"  Max sequence length: {max_seq_len}")

    # Prefill phase: Process initial prompt (e.g., "Hello world")
    prompt_len = 8
    print(f"\n--- Prefill Phase (prompt_len={prompt_len}) ---")

    # Generate random Q, K, V for the prompt
    mx.random.seed(42)
    q_prefill = mx.random.normal((batch_size, prompt_len, num_heads, head_dim)) * 0.1
    k_new = mx.random.normal((batch_size, prompt_len, num_kv_heads, head_dim)) * 0.1
    v_new = mx.random.normal((batch_size, prompt_len, num_kv_heads, head_dim)) * 0.1

    # Run attention with KV cache update
    output = flash_attn_with_kvcache(
        q=q_prefill,
        k_cache=k_cache,
        v_cache=v_cache,
        k=k_new,
        v=v_new,
        cache_seqlens=cache_seqlen,
        causal=True,
        softmax_scale=1.0 / (head_dim ** 0.5),
    )
    mx.eval(output, k_cache, v_cache)

    # Update cache position
    cache_seqlen = prompt_len

    print(f"  Output shape: {output.shape}")
    print(f"  Cache position after prefill: {cache_seqlen}")

    # Decode phase: Generate tokens one at a time
    num_decode_steps = 10
    print(f"\n--- Decode Phase ({num_decode_steps} tokens) ---")

    for step in range(num_decode_steps):
        # Generate Q, K, V for single new token
        q_decode = mx.random.normal((batch_size, 1, num_heads, head_dim)) * 0.1
        k_new_token = mx.random.normal((batch_size, 1, num_kv_heads, head_dim)) * 0.1
        v_new_token = mx.random.normal((batch_size, 1, num_kv_heads, head_dim)) * 0.1

        # Run attention - this attends to all cached KV + new token
        output = flash_attn_with_kvcache(
            q=q_decode,
            k_cache=k_cache,
            v_cache=v_cache,
            k=k_new_token,
            v=v_new_token,
            cache_seqlens=cache_seqlen,
            causal=True,
            softmax_scale=1.0 / (head_dim ** 0.5),
        )
        mx.eval(output, k_cache, v_cache)

        # Update cache position
        cache_seqlen += 1

        if step < 3 or step == num_decode_steps - 1:
            print(f"  Step {step + 1}: Output shape = {output.shape}, "
                  f"Cache position = {cache_seqlen}")
        elif step == 3:
            print("  ...")

    print(f"\n  Final cache utilization: {cache_seqlen}/{max_seq_len} "
          f"({100 * cache_seqlen / max_seq_len:.1f}%)")


def demonstrate_without_new_kv():
    """Show attention using only cached KV (no new tokens)."""
    print("\n" + "=" * 60)
    print("Attention with Pre-populated Cache (no new K/V)")
    print("=" * 60)

    batch_size = 1
    num_heads = 4
    head_dim = 64
    cache_len = 16

    # Pre-populate cache with existing sequence
    mx.random.seed(123)
    k_cache = mx.random.normal((batch_size, cache_len, num_heads, head_dim)) * 0.1
    v_cache = mx.random.normal((batch_size, cache_len, num_heads, head_dim)) * 0.1
    cache_seqlen = cache_len  # All positions are valid

    # Query attends to cached sequence (no new K/V added)
    q = mx.random.normal((batch_size, 1, num_heads, head_dim)) * 0.1

    output = flash_attn_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_seqlens=cache_seqlen,
        causal=False,  # Can attend to all cached positions
    )
    mx.eval(output)

    print(f"\nQuery shape: {q.shape}")
    print(f"Cache shape: {k_cache.shape}")
    print(f"Cache seqlen: {cache_seqlen}")
    print(f"Output shape: {output.shape}")
    print("  (Query attends to all cached positions)")


def demonstrate_gqa_kv_cache():
    """Show GQA with KV cache where num_kv_heads < num_heads."""
    print("\n" + "=" * 60)
    print("GQA with KV Cache (8 Q heads, 2 KV heads)")
    print("=" * 60)

    batch_size = 1
    num_heads = 8
    num_kv_heads = 2  # 4:1 head ratio
    head_dim = 64
    max_seq_len = 64

    # Allocate smaller KV cache (fewer heads)
    k_cache = mx.zeros((batch_size, max_seq_len, num_kv_heads, head_dim))
    v_cache = mx.zeros((batch_size, max_seq_len, num_kv_heads, head_dim))
    cache_seqlen = 0

    # Process a few tokens
    mx.random.seed(456)
    q = mx.random.normal((batch_size, 4, num_heads, head_dim)) * 0.1
    k_new = mx.random.normal((batch_size, 4, num_kv_heads, head_dim)) * 0.1
    v_new = mx.random.normal((batch_size, 4, num_kv_heads, head_dim)) * 0.1

    output = flash_attn_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        k=k_new,
        v=v_new,
        cache_seqlens=cache_seqlen,
        causal=True,
    )
    mx.eval(output, k_cache, v_cache)

    print(f"\nQuery heads: {num_heads}")
    print(f"KV heads: {num_kv_heads}")
    print(f"Head ratio: {num_heads // num_kv_heads}:1")
    print(f"Output shape: {output.shape}")
    print(f"KV cache memory savings: {(1 - num_kv_heads / num_heads) * 100:.0f}%")


if __name__ == "__main__":
    simulate_autoregressive_generation()
    demonstrate_without_new_kv()
    demonstrate_gqa_kv_cache()

    print("\n" + "=" * 60)
    print("All KV cache examples completed successfully!")
    print("=" * 60)

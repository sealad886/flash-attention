#!/usr/bin/env python3
"""
Sliding Window Attention Example

This example demonstrates Flash Attention with sliding window (local) attention,
where each query can only attend to a limited window of keys around it.

Sliding window attention is useful for:
- Very long sequences where full attention is too expensive
- Models like Mistral that use sliding window + full attention hybrid
- Sparse attention patterns

Requirements:
    pip install flash-attn mlx

Window Size Parameter:
    window_size=(left, right) where:
    - left: Number of keys to attend to on the left of current position
    - right: Number of keys to attend to on the right of current position
    - (-1, -1): No window limit (full attention)

Expected output:
    === Full Attention (No Window) ===
    Each query attends to all 2048 keys
    ✅ Full attention computed!

    === Sliding Window Attention ===
    Window size: (128, 128)
    Each query attends to up to 257 keys (128 left + self + 128 right)
    ✅ Sliding window attention computed!

    === Causal Sliding Window ===
    Window size: (128, 0) with causal=True
    Each query attends to up to 129 keys (128 left + self)
    ✅ Causal sliding window computed!
"""

import mlx.core as mx

from flash_attn import flash_attn_func


def main():
    mx.random.seed(42)

    batch_size = 2
    seq_len = 2048
    n_heads = 8
    head_dim = 64

    # Generate inputs
    q = mx.random.normal((batch_size, seq_len, n_heads, head_dim)).astype(mx.float16)
    k = mx.random.normal((batch_size, seq_len, n_heads, head_dim)).astype(mx.float16)
    v = mx.random.normal((batch_size, seq_len, n_heads, head_dim)).astype(mx.float16)

    print(f"Sequence length: {seq_len}")
    print(f"Tensor shapes: Q={q.shape}, K={k.shape}, V={v.shape}\n")

    # Full attention (no window limit)
    print("=== Full Attention (No Window) ===")
    print(f"Each query attends to all {seq_len} keys")
    output_full = flash_attn_func(q, k, v, window_size=(-1, -1))
    mx.eval(output_full)
    print("✅ Full attention computed!\n")

    # Sliding window attention: 128 keys left and right
    print("=== Sliding Window Attention ===")
    window_left, window_right = 128, 128
    print(f"Window size: ({window_left}, {window_right})")
    print(f"Each query attends to up to {window_left + 1 + window_right} keys "
          f"({window_left} left + self + {window_right} right)")
    output_window = flash_attn_func(q, k, v, window_size=(window_left, window_right))
    mx.eval(output_window)
    print("✅ Sliding window attention computed!\n")

    # Causal sliding window: left window only (for autoregressive models)
    print("=== Causal Sliding Window ===")
    window_left = 128
    print(f"Window size: ({window_left}, 0) with causal=True")
    print(f"Each query attends to up to {window_left + 1} keys ({window_left} left + self)")
    output_causal_window = flash_attn_func(
        q, k, v,
        causal=True,
        window_size=(window_left, 0)
    )
    mx.eval(output_causal_window)
    print("✅ Causal sliding window computed!")


if __name__ == "__main__":
    main()

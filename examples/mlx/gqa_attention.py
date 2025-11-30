#!/usr/bin/env python3
"""
Grouped-Query Attention (GQA) and Multi-Query Attention (MQA) Examples

This example demonstrates Flash Attention with different head count ratios
for efficient inference in large language models.

GQA/MQA allows using fewer key-value heads than query heads, reducing
memory usage and computation while maintaining quality.

Requirements:
    pip install flash-attn mlx

Terminology:
    - MQA (Multi-Query Attention): 1 KV head shared across all Q heads
    - GQA (Grouped-Query Attention): Groups of Q heads share KV heads

Expected output:
    === Multi-Query Attention (MQA) ===
    Q heads: 8, KV heads: 1, ratio: 8
    Query shape: (2, 512, 8, 64)
    Key shape: (2, 512, 1, 64)
    Output shape: (2, 512, 8, 64)
    ✅ MQA computed successfully!

    === Grouped-Query Attention (GQA) ===
    Q heads: 8, KV heads: 2, ratio: 4
    Query shape: (2, 512, 8, 64)
    Key shape: (2, 512, 2, 64)
    Output shape: (2, 512, 8, 64)
    ✅ GQA computed successfully!

    === Standard Multi-Head Attention ===
    Q heads: 8, KV heads: 8, ratio: 1
    ✅ Standard MHA computed successfully!
"""

import mlx.core as mx

from flash_attn import flash_attn_func


def run_attention_with_ratio(batch_size: int, seq_len: int, head_dim: int,
                              n_heads_q: int, n_heads_kv: int):
    """Run attention with specified head counts."""
    q = mx.random.normal((batch_size, seq_len, n_heads_q, head_dim)).astype(mx.float16)
    k = mx.random.normal((batch_size, seq_len, n_heads_kv, head_dim)).astype(mx.float16)
    v = mx.random.normal((batch_size, seq_len, n_heads_kv, head_dim)).astype(mx.float16)

    print(f"Query shape: {q.shape}")
    print(f"Key shape: {k.shape}")

    # Flash Attention automatically handles GQA/MQA
    output = flash_attn_func(q, k, v, causal=True)
    mx.eval(output)

    print(f"Output shape: {output.shape}")
    return output


def main():
    mx.random.seed(42)

    batch_size = 2
    seq_len = 512
    head_dim = 64

    # Multi-Query Attention (MQA): 1 KV head for all Q heads
    print("=== Multi-Query Attention (MQA) ===")
    n_heads_q, n_heads_kv = 8, 1
    print(f"Q heads: {n_heads_q}, KV heads: {n_heads_kv}, ratio: {n_heads_q // n_heads_kv}")
    run_attention_with_ratio(batch_size, seq_len, head_dim, n_heads_q, n_heads_kv)
    print("✅ MQA computed successfully!\n")

    # Grouped-Query Attention (GQA): Groups of Q heads share KV heads
    print("=== Grouped-Query Attention (GQA) ===")
    n_heads_q, n_heads_kv = 8, 2
    print(f"Q heads: {n_heads_q}, KV heads: {n_heads_kv}, ratio: {n_heads_q // n_heads_kv}")
    run_attention_with_ratio(batch_size, seq_len, head_dim, n_heads_q, n_heads_kv)
    print("✅ GQA computed successfully!\n")

    # Standard Multi-Head Attention (equal heads)
    print("=== Standard Multi-Head Attention ===")
    n_heads_q, n_heads_kv = 8, 8
    print(f"Q heads: {n_heads_q}, KV heads: {n_heads_kv}, ratio: {n_heads_q // n_heads_kv}")
    run_attention_with_ratio(batch_size, seq_len, head_dim, n_heads_q, n_heads_kv)
    print("✅ Standard MHA computed successfully!")


if __name__ == "__main__":
    main()

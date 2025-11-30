#!/usr/bin/env python3
"""
Packed QKV Variants Example

This example demonstrates the packed tensor variants of Flash Attention:
- flash_attn_qkvpacked_func: For self-attention with QKV packed together
- flash_attn_kvpacked_func: For cross-attention with KV packed together

Packed variants can be more memory-efficient when Q, K, V come from the
same linear projection (as in standard transformer self-attention).

Requirements:
    pip install flash-attn mlx

Tensor Shapes:
    - QKV packed: (batch, seqlen, 3, nheads, headdim)
    - KV packed: (batch, seqlen, 2, nheads, headdim)

Expected output:
    === QKV-Packed Self-Attention ===
    QKV packed shape: (2, 512, 3, 8, 64)
    Output shape: (2, 512, 8, 64)
    ✅ QKV-packed attention computed!

    === KV-Packed Cross-Attention ===
    Query shape: (2, 128, 8, 64)
    KV packed shape: (2, 512, 2, 8, 64)
    Output shape: (2, 128, 8, 64)
    ✅ KV-packed attention computed!
"""

import mlx.core as mx

from flash_attn import flash_attn_qkvpacked_func, flash_attn_kvpacked_func


def main():
    mx.random.seed(42)

    batch_size = 2
    n_heads = 8
    head_dim = 64

    # === QKV-Packed Self-Attention ===
    print("=== QKV-Packed Self-Attention ===")
    seq_len = 512

    # Create QKV packed tensor: (batch, seqlen, 3, nheads, headdim)
    # Index 0 = Q, Index 1 = K, Index 2 = V
    qkv = mx.random.normal((batch_size, seq_len, 3, n_heads, head_dim)).astype(mx.float16)
    print(f"QKV packed shape: {qkv.shape}")

    output_qkv = flash_attn_qkvpacked_func(qkv, causal=True)
    mx.eval(output_qkv)
    print(f"Output shape: {output_qkv.shape}")
    print("✅ QKV-packed attention computed!\n")

    # === KV-Packed Cross-Attention ===
    print("=== KV-Packed Cross-Attention ===")
    seqlen_q = 128   # Query sequence (e.g., decoder)
    seqlen_kv = 512  # Key-value sequence (e.g., encoder output)

    # Separate query tensor
    q = mx.random.normal((batch_size, seqlen_q, n_heads, head_dim)).astype(mx.float16)
    print(f"Query shape: {q.shape}")

    # KV packed tensor: (batch, seqlen_kv, 2, nheads, headdim)
    # Index 0 = K, Index 1 = V
    kv = mx.random.normal((batch_size, seqlen_kv, 2, n_heads, head_dim)).astype(mx.float16)
    print(f"KV packed shape: {kv.shape}")

    # Cross-attention: Q attends to KV (no causal mask for cross-attention)
    output_kv = flash_attn_kvpacked_func(q, kv, causal=False)
    mx.eval(output_kv)
    print(f"Output shape: {output_kv.shape}")
    print("✅ KV-packed attention computed!")


if __name__ == "__main__":
    main()

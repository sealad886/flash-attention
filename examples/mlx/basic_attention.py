#!/usr/bin/env python3
"""
Basic Flash Attention Usage on Apple Silicon

This example demonstrates basic usage of Flash Attention with the MLX backend
on Apple Silicon devices. The same code works on NVIDIA/AMD GPUs with the
CUDA/ROCm backends - just `pip install flash-attn` and import.

Requirements:
    pip install flash-attn mlx

Expected output (shapes will vary based on your parameters):
    Query shape: (2, 1024, 8, 64)
    Key shape: (2, 1024, 8, 64)
    Value shape: (2, 1024, 8, 64)
    Output shape: (2, 1024, 8, 64)
    ✅ Basic attention computed successfully!
    ✅ Causal attention computed successfully!
    ✅ Custom scale attention computed successfully!
"""

import mlx.core as mx

# Platform-aware import: works on Apple Silicon, NVIDIA, and AMD
from flash_attn import flash_attn_func


def main():
    # Create sample input tensors
    # Shape: (batch, seqlen, nheads, headdim)
    batch_size = 2
    seq_len = 1024
    n_heads = 8
    head_dim = 64

    # Generate random Q, K, V tensors
    mx.random.seed(42)
    q = mx.random.normal((batch_size, seq_len, n_heads, head_dim)).astype(mx.float16)
    k = mx.random.normal((batch_size, seq_len, n_heads, head_dim)).astype(mx.float16)
    v = mx.random.normal((batch_size, seq_len, n_heads, head_dim)).astype(mx.float16)

    print(f"Query shape: {q.shape}")
    print(f"Key shape: {k.shape}")
    print(f"Value shape: {v.shape}")

    # Basic attention (no causal masking)
    output = flash_attn_func(q, k, v)
    mx.eval(output)  # Force evaluation for MLX lazy execution
    print(f"Output shape: {output.shape}")
    print("✅ Basic attention computed successfully!")

    # Causal attention (for autoregressive models like GPT)
    output_causal = flash_attn_func(q, k, v, causal=True)
    mx.eval(output_causal)
    print("✅ Causal attention computed successfully!")

    # Custom softmax scale
    custom_scale = 0.1
    output_scaled = flash_attn_func(q, k, v, softmax_scale=custom_scale)
    mx.eval(output_scaled)
    print("✅ Custom scale attention computed successfully!")


if __name__ == "__main__":
    main()

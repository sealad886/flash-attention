#!/usr/bin/env python3
"""
Training with Gradients Example

This example demonstrates using Flash Attention in a training loop with
automatic differentiation using MLX's grad system.

This shows:
- Forward pass through attention
- Computing loss
- Backward pass to get gradients
- Simple gradient descent update

Requirements:
    pip install flash-attn mlx

Expected output:
    === Training Loop Example ===
    Initial output shape: (1, 256, 4, 64)
    Step 0: Loss = X.XXXX
    Step 1: Loss = X.XXXX
    ...
    ✅ Training loop completed successfully!
"""

import mlx.core as mx

from flash_attn import flash_attn_func


def attention_loss_fn(q: mx.array, k: mx.array, v: mx.array, target: mx.array) -> mx.array:
    """Compute MSE loss between attention output and target."""
    output = flash_attn_func(q, k, v, causal=True)
    loss = mx.mean((output - target) ** 2)
    return loss


def main():
    mx.random.seed(42)

    batch_size = 1
    seq_len = 256
    n_heads = 4
    head_dim = 64
    learning_rate = 0.01
    n_steps = 5

    print("=== Training Loop Example ===")

    # Initialize "trainable" Q, K, V (in practice these come from linear projections)
    q = mx.random.normal((batch_size, seq_len, n_heads, head_dim)).astype(mx.float32)
    k = mx.random.normal((batch_size, seq_len, n_heads, head_dim)).astype(mx.float32)
    v = mx.random.normal((batch_size, seq_len, n_heads, head_dim)).astype(mx.float32)

    # Random target (in practice this would be your actual target)
    target = mx.random.normal((batch_size, seq_len, n_heads, head_dim)).astype(mx.float32)

    # Initial forward pass
    output = flash_attn_func(q, k, v, causal=True)
    mx.eval(output)
    print(f"Initial output shape: {output.shape}")

    # Training loop with gradient descent on Q (as a simple example)
    grad_fn = mx.grad(attention_loss_fn, argnums=0)  # Gradient w.r.t. Q

    for step in range(n_steps):
        # Compute loss
        loss = attention_loss_fn(q, k, v, target)

        # Compute gradient of loss w.r.t. Q
        dq = grad_fn(q, k, v, target)

        # Evaluate to get actual values
        mx.eval(loss, dq)

        print(f"Step {step}: Loss = {loss.item():.4f}")

        # Simple gradient descent update
        q = q - learning_rate * dq

    print("✅ Training loop completed successfully!")


if __name__ == "__main__":
    main()

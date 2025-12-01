#!/usr/bin/env python3
"""Quick MLX smoke test for Flash Attention."""

import sys


def main():
    try:
        from flash_attn.flash_attn_mlx import flash_attn_func, is_mlx_available

        if not is_mlx_available():
            print("⚠️  MLX not available (no Metal GPU)")
            return 0

        import mlx.core as mx

        # Quick smoke test with small tensors
        q = mx.random.normal((2, 128, 8, 64))
        k = mx.random.normal((2, 128, 8, 64))
        v = mx.random.normal((2, 128, 8, 64))

        out = flash_attn_func(q, k, v)
        mx.eval(out)

        assert out.shape == q.shape, f"Shape mismatch: {out.shape} vs {q.shape}"

        # Test causal attention
        out_causal = flash_attn_func(q, k, v, causal=True)
        mx.eval(out_causal)

        print("✅ MLX smoke test passed!")
        return 0

    except ImportError as e:
        print(f"⚠️  MLX not installed: {e}")
        return 0

    except Exception as e:
        print(f"❌ MLX smoke test failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

# MLX Flash Attention Examples

This directory contains examples demonstrating Flash Attention usage on Apple Silicon with the MLX backend.

## Prerequisites

```bash
# Install flash-attn (works on Apple Silicon, NVIDIA, and AMD)
pip install flash-attn

# On Apple Silicon, you'll also need MLX
pip install mlx
```

## Examples

### 1. Basic Attention (`basic_attention.py`)

Simple example showing:
- Standard attention computation
- Causal masking for autoregressive models
- Custom softmax scaling

```bash
python basic_attention.py
```

### 2. GQA/MQA Attention (`gqa_attention.py`)

Demonstrates Grouped-Query and Multi-Query Attention:
- MQA: Single KV head shared across all Q heads
- GQA: Groups of Q heads share KV heads
- Standard MHA: Equal Q and KV heads

```bash
python gqa_attention.py
```

### 3. Sliding Window Attention (`sliding_window_attention.py`)

Shows local/sparse attention patterns:
- Full attention (no window)
- Symmetric sliding window
- Causal sliding window for autoregressive models

```bash
python sliding_window_attention.py
```

### 4. Training with Gradients (`training_with_gradients.py`)

Example of using Flash Attention in a training loop:
- Forward pass computation
- Loss calculation
- Automatic differentiation with MLX's grad
- Gradient descent updates

```bash
python training_with_gradients.py
```

### 5. Packed Variants (`packed_variants.py`)

Efficient packed tensor variants:
- QKV-packed for self-attention
- KV-packed for cross-attention

```bash
python packed_variants.py
```

### 6. KV Cache Inference (`kv_cache_inference.py`)

Efficient autoregressive inference with KV caching:
- Prefill phase: Process initial prompt
- Decode phase: Generate tokens one at a time
- GQA with KV cache for memory savings
- Using pre-populated cache without new K/V

```bash
python kv_cache_inference.py
```

## Platform Compatibility

The same `flash_attn` API works across platforms:

```python
# This import works on Apple Silicon, NVIDIA, and AMD
from flash_attn import flash_attn_func

# Use the same API regardless of backend
output = flash_attn_func(q, k, v, causal=True)
```

The backend is automatically selected based on your hardware:
- **Apple Silicon (M1/M2/M3)**: Uses MLX backend with Metal
- **NVIDIA GPUs**: Uses CUDA backend
- **AMD GPUs**: Uses ROCm/Triton backend

## Supported Features

| Feature | Status |
|---------|--------|
| Basic attention | ✅ |
| Causal masking | ✅ |
| Custom softmax scale | ✅ |
| GQA/MQA | ✅ |
| Sliding window | ✅ |
| Backward pass (gradients) | ✅ |
| QKV/KV packed | ✅ |
| KV cache inference | ✅ |
| Softcap | ✅ |
| ALiBi (via fallback) | ✅ |
| Dropout (via fallback) | ✅ |
| FP16/FP32 | ✅ |

## Performance Notes

Flash Attention on Apple Silicon leverages:
- **Unified Memory**: No CPU-GPU data transfers needed
- **Metal Compute**: Optimized Metal shaders for attention
- **Online Softmax**: Memory-efficient algorithm that doesn't materialize full attention matrix

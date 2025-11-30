"""Tests for Flash Attention MLX Backend.

This test module validates the MLX backend implementation against
a reference implementation. Tests are designed to match the patterns
in tests/test_flash_attn.py for consistency.

These tests require macOS with Apple Silicon (M1/M2/M3) and MLX installed.
"""

from __future__ import annotations

import math
import sys
from typing import TYPE_CHECKING, Any, Callable, Dict, Sequence, Tuple, Union

import pytest

from flash_attn.flash_attn_mlx.reference import varlen_attention_ref_mlx
from flash_attn.flash_attn_mlx.varlen_bwd_kernel import (
    varlen_flash_attention_backward,
)
from flash_attn.flash_attn_mlx.tests.varlen_test_utils import (
    VARLEN_CAUSAL_FLAGS,
    VARLEN_HEAD_CONFIGS,
    VARLEN_HEADDIMS,
    VARLEN_SEQLEN_CASES,
    generate_cu_seqlens,
    make_varlen_qkv,
)
from flash_attn.flash_attn_mlx.utils import USE_REF

# Skip all tests if not on Apple Silicon or MLX unavailable
pytestmark = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="MLX tests require macOS"
)


# ============================================================================
# Test Fixtures and Helpers
# ============================================================================

if TYPE_CHECKING:
    import mlx.core as mx

    # Type aliases for MLX module and arrays
    MXModule = Any  # mlx.core module
    MXArray = mx.array
    FlashAttnFunc = Callable[..., Union[MXArray, Tuple[MXArray, MXArray, MXArray]]]
    AttentionRefFunc = Callable[..., Tuple[MXArray, MXArray, Any]]


@pytest.fixture(scope="module")
def mlx() -> Any:
    """Import MLX, skip if unavailable."""
    pytest.importorskip("mlx")
    import mlx.core as mx
    return mx


@pytest.fixture(scope="module")
def flash_attn_mlx(mlx: Any) -> Callable[..., Any]:
    """Import Flash Attention MLX backend."""
    from flash_attn.flash_attn_mlx import flash_attn_func
    return flash_attn_func


@pytest.fixture(scope="module")
def flash_attn_varlen_api(mlx: Any) -> Callable[..., Any]:
    """Import the varlen Flash Attention entry point."""
    from flash_attn.flash_attn_mlx import flash_attn_varlen_func

    return flash_attn_varlen_func


@pytest.fixture(scope="module")
def attention_ref(mlx: Any) -> Callable[..., Tuple[Any, Any, Any]]:
    """Import reference implementation."""
    from flash_attn.flash_attn_mlx.reference import attention_ref_mlx
    return attention_ref_mlx


@pytest.fixture(scope="module")
def device_info(mlx: Any) -> Dict[str, Any]:
    """Get device information."""
    from flash_attn.flash_attn_mlx.device import get_device_info
    return get_device_info()


def make_qkv(
    mx: Any,
    batch: int,
    seqlen_q: int,
    seqlen_k: int,
    nheads: int,
    nheads_k: int,
    headdim: int,
    dtype: Any = None,
    seed: int = 42,
) -> Tuple[Any, Any, Any]:
    """
    Create random Q, K, V tensors for testing.

    Args:
        mx: MLX module
        batch: Batch size
        seqlen_q: Query sequence length
        seqlen_k: Key/value sequence length
        nheads: Number of query heads
        nheads_k: Number of key/value heads
        headdim: Head dimension
        dtype: Data type (default: float16)
        seed: Random seed

    Returns:
        Tuple of (q, k, v) arrays
    """
    if dtype is None:
        dtype = mx.float16

    mx.random.seed(seed)

    q = mx.random.normal((batch, seqlen_q, nheads, headdim)).astype(dtype)
    k = mx.random.normal((batch, seqlen_k, nheads_k, headdim)).astype(dtype)
    v = mx.random.normal((batch, seqlen_k, nheads_k, headdim)).astype(dtype)

    # Scale down to avoid numerical issues
    q = q * 0.1
    k = k * 0.1
    v = v * 0.1

    return q, k, v


def _seqlens_id(seqlens: Sequence[int]) -> str:
    """Generate a test ID string from sequence lengths."""
    return "-".join(str(int(length)) for length in seqlens)


def assert_close(
    mx: Any,
    actual: Any,
    expected: Any,
    atol: float = 1e-3,
    rtol: float = 1e-3,
    msg: str = "",
) -> None:
    """
    Assert that two arrays are close within tolerance.

    Args:
        mx: MLX module
        actual: Actual array
        expected: Expected array
        atol: Absolute tolerance
        rtol: Relative tolerance
        msg: Optional message for assertion error

    Raises:
        pytest.Failed: If arrays differ beyond tolerance
    """
    # Evaluate arrays if lazy
    actual = mx.eval(actual)
    expected = mx.eval(expected)

    diff = mx.abs(actual - expected)
    max_diff = float(mx.max(diff))

    # Compute relative difference
    scale = mx.maximum(mx.abs(expected), mx.array(1e-8))
    rel_diff = diff / scale
    max_rel_diff = float(mx.max(rel_diff))

    if max_diff > atol or max_rel_diff > rtol:
        pytest.fail(
            f"{msg}\nMax absolute difference: {max_diff:.6f} (tolerance: {atol})\n"
            f"Max relative difference: {max_rel_diff:.6f} (tolerance: {rtol})"
        )


# ============================================================================
# Device Detection Tests
# ============================================================================

class TestDeviceDetection:
    """Test device detection utilities."""

    def test_is_apple_silicon(self, mlx: Any) -> None:
        """Test Apple Silicon detection."""
        from flash_attn.flash_attn_mlx.device import is_apple_silicon
        # On macOS ARM64, this should be True
        if sys.platform == "darwin":
            import platform
            expected = platform.machine() == "arm64"
            assert is_apple_silicon() == expected

    def test_is_mlx_available(self, mlx: Any) -> None:
        """Test MLX availability check."""
        from flash_attn.flash_attn_mlx.device import is_mlx_available
        # If we got here, MLX is available
        assert is_mlx_available() is True

    def test_get_gpu_family(self, mlx: Any) -> None:
        """Test GPU family detection."""
        from flash_attn.flash_attn_mlx.device import get_gpu_family
        family = get_gpu_family()
        assert family in ("apple7", "apple8", "apple9", "unknown")

    def test_get_block_sizes(self, mlx: Any) -> None:
        """Test block size heuristics."""
        from flash_attn.flash_attn_mlx.device import get_block_sizes

        for headdim in [32, 64, 96, 128]:
            block_m, block_n = get_block_sizes(headdim)
            assert block_m > 0
            assert block_n > 0
            assert block_m % 16 == 0  # Should be multiple of SIMD width
            assert block_n % 16 == 0

    def test_device_info(self, device_info: Dict[str, Any]) -> None:
        """Test device info dictionary."""
        assert "platform" in device_info
        assert "mlx_available" in device_info
        assert device_info["mlx_available"] is True


# ============================================================================
# Reference Implementation Tests
# ============================================================================

class TestReferenceImplementation:
    """Test the reference attention implementation."""

    @pytest.mark.parametrize("headdim", [32, 64, 128])
    @pytest.mark.parametrize("causal", [False, True])
    def test_reference_basic(
        self, mlx: Any, attention_ref: Callable[..., Any], headdim: int, causal: bool
    ) -> None:
        """Test basic attention computation."""
        batch, seqlen, nheads = 2, 64, 4
        q, k, v = make_qkv(mlx, batch, seqlen, seqlen, nheads, nheads, headdim)

        out, lse, _ = attention_ref(q, k, v, causal=causal)

        # Output should have same shape as Q
        assert out.shape == q.shape

        # LSE should have shape (batch, nheads, seqlen)
        assert lse.shape == (batch, nheads, seqlen)

        # Output should not be all zeros or NaN
        out = mlx.eval(out)
        assert not mlx.all(out == 0)
        assert not mlx.any(mlx.isnan(out))

    @pytest.mark.parametrize("nheads,nheads_k", [(8, 8), (8, 4), (8, 2), (8, 1)])
    def test_reference_gqa(
        self, mlx: Any, attention_ref: Callable[..., Any], nheads: int, nheads_k: int
    ) -> None:
        """Test grouped-query attention (GQA/MQA)."""
        batch, seqlen, headdim = 2, 64, 64
        q, k, v = make_qkv(mlx, batch, seqlen, seqlen, nheads, nheads_k, headdim)

        out, lse, _ = attention_ref(q, k, v, causal=True)

        assert out.shape == q.shape
        out = mlx.eval(out)
        assert not mlx.any(mlx.isnan(out))

    def test_reference_cross_attention(
        self, mlx: Any, attention_ref: Callable[..., Any]
    ) -> None:
        """Test cross attention with different Q and K/V lengths."""
        batch, seqlen_q, seqlen_k = 2, 32, 64
        nheads, headdim = 4, 64
        q, k, v = make_qkv(mlx, batch, seqlen_q, seqlen_k, nheads, nheads, headdim)

        # Non-causal cross attention
        out, lse, _ = attention_ref(q, k, v, causal=False)

        assert out.shape == (batch, seqlen_q, nheads, headdim)
        out = mlx.eval(out)
        assert not mlx.any(mlx.isnan(out))

    def test_reference_softmax_scale(
        self, mlx: Any, attention_ref: Callable[..., Any]
    ) -> None:
        """Test custom softmax scale."""
        batch, seqlen, nheads, headdim = 2, 64, 4, 64
        q, k, v = make_qkv(mlx, batch, seqlen, seqlen, nheads, nheads, headdim)

        # Default scale
        out1, _, _ = attention_ref(q, k, v)

        # Custom scale
        scale = 0.5 / math.sqrt(headdim)
        out2, _, _ = attention_ref(q, k, v, softmax_scale=scale)

        # Outputs should be different
        out1, out2 = mlx.eval(out1), mlx.eval(out2)
        assert not mlx.allclose(out1, out2)


# ============================================================================
# Flash Attention Function Tests
# ============================================================================

class TestFlashAttnFunc:
    """Test the main flash_attn_func."""

    @pytest.mark.parametrize("headdim", [32, 64, 128])
    @pytest.mark.parametrize("causal", [False, True])
    def test_flash_attn_basic(
        self,
        mlx: Any,
        flash_attn_mlx: Callable[..., Any],
        attention_ref: Callable[..., Any],
        headdim: int,
        causal: bool,
    ) -> None:
        """Test basic flash attention matches reference."""
        batch, seqlen, nheads = 2, 64, 4
        q, k, v = make_qkv(mlx, batch, seqlen, seqlen, nheads, nheads, headdim)

        # Flash attention
        out_flash = flash_attn_mlx(q, k, v, causal=causal)

        # Reference
        out_ref, _, _ = attention_ref(q, k, v, causal=causal)

        # Compare
        assert_close(mlx, out_flash, out_ref, atol=1e-2, rtol=1e-2,
                     msg=f"Flash attention mismatch (headdim={headdim}, causal={causal})")

    @pytest.mark.parametrize("batch", [1, 2, 4])
    def test_flash_attn_batch_sizes(
        self,
        mlx: Any,
        flash_attn_mlx: Callable[..., Any],
        attention_ref: Callable[..., Any],
        batch: int,
    ) -> None:
        """Test various batch sizes."""
        seqlen, nheads, headdim = 64, 4, 64
        q, k, v = make_qkv(mlx, batch, seqlen, seqlen, nheads, nheads, headdim)

        out_flash = flash_attn_mlx(q, k, v, causal=True)
        out_ref, _, _ = attention_ref(q, k, v, causal=True)

        assert_close(mlx, out_flash, out_ref, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("seqlen", [32, 64, 128, 256])
    def test_flash_attn_sequence_lengths(
        self,
        mlx: Any,
        flash_attn_mlx: Callable[..., Any],
        attention_ref: Callable[..., Any],
        seqlen: int,
    ) -> None:
        """Test various sequence lengths."""
        batch, nheads, headdim = 2, 4, 64
        q, k, v = make_qkv(mlx, batch, seqlen, seqlen, nheads, nheads, headdim)

        out_flash = flash_attn_mlx(q, k, v, causal=True)
        out_ref, _, _ = attention_ref(q, k, v, causal=True)

        assert_close(mlx, out_flash, out_ref, atol=1e-2, rtol=1e-2)

    def test_flash_attn_return_attn_probs(
        self, mlx: Any, flash_attn_mlx: Callable[..., Any]
    ) -> None:
        """Test returning attention probabilities."""
        batch, seqlen, nheads, headdim = 2, 64, 4, 64
        q, k, v = make_qkv(mlx, batch, seqlen, seqlen, nheads, nheads, headdim)

        result = flash_attn_mlx(q, k, v, causal=True, return_attn_probs=True)

        assert isinstance(result, tuple)
        assert len(result) == 3  # out, softmax_lse, attn_probs


# ============================================================================
# Packed Variants Tests
# ============================================================================

class TestPackedVariants:
    """Test QKV-packed and KV-packed variants."""

    def test_qkvpacked(self, mlx: Any) -> None:
        """Test flash_attn_qkvpacked_func."""
        from flash_attn.flash_attn_mlx import flash_attn_qkvpacked_func

        batch, seqlen, nheads, headdim = 2, 64, 4, 64

        # Create packed QKV tensor
        mx = mlx
        mx.random.seed(42)
        qkv = mx.random.normal((batch, seqlen, 3, nheads, headdim)).astype(mx.float16) * 0.1

        out = flash_attn_qkvpacked_func(qkv, causal=True)

        assert out.shape == (batch, seqlen, nheads, headdim)
        out = mx.eval(out)
        assert not mx.any(mx.isnan(out))

    def test_kvpacked(self, mlx: Any) -> None:
        """Test flash_attn_kvpacked_func."""
        from flash_attn.flash_attn_mlx import flash_attn_kvpacked_func

        batch, seqlen_q, seqlen_k, nheads, headdim = 2, 32, 64, 4, 64

        mx = mlx
        mx.random.seed(42)
        q = mx.random.normal((batch, seqlen_q, nheads, headdim)).astype(mx.float16) * 0.1
        kv = mx.random.normal((batch, seqlen_k, 2, nheads, headdim)).astype(mx.float16) * 0.1

        out = flash_attn_kvpacked_func(q, kv, causal=False)

        assert out.shape == (batch, seqlen_q, nheads, headdim)
        out = mx.eval(out)
        assert not mx.any(mx.isnan(out))


@pytest.mark.skipif(USE_REF, reason="Varlen tests require Metal kernels (set FLASH_ATTENTION_MLX_REF=0)")
class TestVarlenAPI:
    """End-to-end tests for the public varlen MLX API."""

    _common_seqlens = VARLEN_SEQLEN_CASES[:4]
    _edge_seqlens = VARLEN_SEQLEN_CASES[4:]

    @pytest.mark.parametrize("seqlens", _common_seqlens, ids=_seqlens_id)
    @pytest.mark.parametrize("causal", VARLEN_CAUSAL_FLAGS)
    @pytest.mark.parametrize("headdim", VARLEN_HEADDIMS)
    @pytest.mark.parametrize("head_config", VARLEN_HEAD_CONFIGS)
    def test_varlen_forward_matches_reference(
        self,
        flash_attn_varlen_api: Callable[..., Any],
        mlx: Any,
        seqlens: Sequence[int],
        causal: bool,
        headdim: int,
        head_config: Tuple[int, int],
    ) -> None:
        """flash_attn_varlen_func output should match the reference implementation."""

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

        out, lse, _ = flash_attn_varlen_api(
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            causal=causal,
            return_attn_probs=True,
        )

        ref_out, ref_lse, _ = varlen_attention_ref_mlx(
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            causal=causal,
        )

        mx_mod = mlx
        diff_out = float(mx_mod.max(mx_mod.abs(out - ref_out)))
        diff_lse = float(mx_mod.max(mx_mod.abs(lse - ref_lse)))

        assert diff_out < 1e-3, (
            f"Varlen API mismatch (seqlens={seqlens}, headdim={headdim}, "
            f"nheads={nheads}, nheads_k={nheads_k}, causal={causal}): {diff_out}"
        )
        assert diff_lse < 5e-3, f"Varlen API LSE mismatch: {diff_lse}"

    @pytest.mark.parametrize("seqlens", _common_seqlens[:2], ids=_seqlens_id)
    @pytest.mark.parametrize("causal", VARLEN_CAUSAL_FLAGS)
    @pytest.mark.parametrize("head_config", VARLEN_HEAD_CONFIGS)
    def test_varlen_backward_autograd(
        self,
        flash_attn_varlen_api: Callable[..., Any],
        mlx: Any,
        seqlens: Sequence[int],
        causal: bool,
        head_config: Tuple[int, int],
    ) -> None:
        """Gradients from mx.grad should agree with reference varlen backward."""

        nheads, nheads_k = head_config
        headdim = 64
        mx_mod = mlx

        (
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
        ) = make_varlen_qkv(seqlens, nheads, nheads_k, headdim)

        mx_mod.random.seed(17)
        upstream = mx_mod.random.normal(q_varlen.shape).astype(q_varlen.dtype) * 0.02

        def loss_fn(q_in, k_in, v_in):
            out = flash_attn_varlen_api(
                q_in,
                k_in,
                v_in,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                causal=causal,
            )
            return mx_mod.sum(out * upstream)

        dq_auto, dk_auto, dv_auto = mx_mod.grad(loss_fn, argnums=(0, 1, 2))(
            q_varlen,
            k_varlen,
            v_varlen,
        )

        out_kernel, lse_kernel, _ = flash_attn_varlen_api(
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            causal=causal,
            return_attn_probs=True,
        )

        dq_ref, dk_ref, dv_ref = varlen_flash_attention_backward(
            q_varlen,
            k_varlen,
            v_varlen,
            out_kernel,
            upstream,
            lse_kernel,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=causal,
            use_metal_kernel=False,
        )

        diff_dq = float(mx_mod.max(mx_mod.abs(dq_auto - dq_ref)))
        diff_dk = float(mx_mod.max(mx_mod.abs(dk_auto - dk_ref)))
        diff_dv = float(mx_mod.max(mx_mod.abs(dv_auto - dv_ref)))

        assert diff_dq < 5e-2, f"Varlen dq mismatch: {diff_dq}"
        assert diff_dk < 5e-2, f"Varlen dk mismatch: {diff_dk}"
        assert diff_dv < 5e-2, f"Varlen dv mismatch: {diff_dv}"

    @pytest.mark.parametrize("seqlens", _edge_seqlens, ids=_seqlens_id)
    @pytest.mark.parametrize("causal", VARLEN_CAUSAL_FLAGS)
    def test_varlen_edge_sequences(
        self,
        flash_attn_varlen_api: Callable[..., Any],
        mlx: Any,
        seqlens: Sequence[int],
        causal: bool,
    ) -> None:
        """Cover extreme lengths like all-ones and very long sequences."""

        nheads, nheads_k, headdim = 8, 2, 64

        (
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
        ) = make_varlen_qkv(seqlens, nheads, nheads_k, headdim)

        out = flash_attn_varlen_api(
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            causal=causal,
        )

        ref_out, _, _ = varlen_attention_ref_mlx(
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            causal=causal,
        )

        mx_mod = mlx
        diff = float(mx_mod.max(mx_mod.abs(out - ref_out)))
        assert diff < 1e-3, f"Edge varlen mismatch ({seqlens}, causal={causal}): {diff}"

    @pytest.mark.parametrize("seqlens", _common_seqlens[:3], ids=_seqlens_id)
    @pytest.mark.parametrize("causal", VARLEN_CAUSAL_FLAGS)
    def test_varlen_packed_variants(
        self,
        flash_attn_varlen_api: Callable[..., Any],
        mlx: Any,
        seqlens: Sequence[int],
        causal: bool,
    ) -> None:
        """Public qkvpacked/kvpacked helpers should align with base varlen output."""

        from flash_attn.flash_attn_mlx import (
            flash_attn_varlen_kvpacked_func,
            flash_attn_varlen_qkvpacked_func,
        )

        nheads, nheads_k, headdim = 8, 2, 64

        # QKV-packed helper expects the same number of heads for Q/K/V.
        (
            q_mha,
            k_mha,
            v_mha,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
        ) = make_varlen_qkv(seqlens, nheads, nheads, headdim)

        baseline_mha = flash_attn_varlen_api(
            q_mha,
            k_mha,
            v_mha,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            causal=causal,
        )
        if isinstance(baseline_mha, tuple):
            baseline_mha = baseline_mha[0]

        mx_mod = mlx
        qkv_packed = mx_mod.stack([q_mha, k_mha, v_mha], axis=1)
        packed_qkv = flash_attn_varlen_qkvpacked_func(
            qkv_packed,
            cu_seqlens_q,
            max_seqlen_q,
            causal=causal,
        )
        if isinstance(packed_qkv, tuple):
            packed_qkv = packed_qkv[0]

        diff_qkv = float(mx_mod.max(mx_mod.abs(baseline_mha - packed_qkv)))
        assert diff_qkv < 1e-3, f"Varlen qkvpacked mismatch: {diff_qkv}"

        # KV-packed helper must support GQA/MQA where nheads != nheads_k.
        (
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
        ) = make_varlen_qkv(seqlens, nheads, nheads_k, headdim)

        baseline_gqa = flash_attn_varlen_api(
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            causal=causal,
        )
        if isinstance(baseline_gqa, tuple):
            baseline_gqa = baseline_gqa[0]

        kv_packed = mx_mod.stack([k_varlen, v_varlen], axis=1)
        packed_kv = flash_attn_varlen_kvpacked_func(
            q_varlen,
            kv_packed,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            causal=causal,
        )
        if isinstance(packed_kv, tuple):
            packed_kv = packed_kv[0]

        diff_kv = float(mx_mod.max(mx_mod.abs(baseline_gqa - packed_kv)))
        assert diff_kv < 1e-3, f"Varlen kvpacked mismatch: {diff_kv}"


# ============================================================================
# Data Type Tests
# ============================================================================

class TestDataTypes:
    """Test different data types."""

    def test_float16(
        self, mlx: Any, flash_attn_mlx: Callable[..., Any], attention_ref: Callable[..., Any]
    ) -> None:
        """Test float16 inputs."""
        batch, seqlen, nheads, headdim = 2, 64, 4, 64
        q, k, v = make_qkv(mlx, batch, seqlen, seqlen, nheads, nheads, headdim,
                          dtype=mlx.float16)

        out = flash_attn_mlx(q, k, v, causal=True)

        assert out.dtype == mlx.float16
        out = mlx.eval(out)
        assert not mlx.any(mlx.isnan(out))

    def test_bfloat16(self, mlx: Any, flash_attn_mlx: Callable[..., Any]) -> None:
        """Test bfloat16 inputs (if supported)."""
        from flash_attn.flash_attn_mlx.device import get_metal_capabilities

        caps = get_metal_capabilities()
        if not caps.get("supports_bfloat16", False):
            pytest.skip("BFloat16 not supported on this device")

        batch, seqlen, nheads, headdim = 2, 64, 4, 64
        q, k, v = make_qkv(mlx, batch, seqlen, seqlen, nheads, nheads, headdim,
                          dtype=mlx.bfloat16)

        out = flash_attn_mlx(q, k, v, causal=True)

        assert out.dtype == mlx.bfloat16
        out = mlx.eval(out)
        assert not mlx.any(mlx.isnan(out))


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_token(self, mlx: Any, flash_attn_mlx: Callable[..., Any]) -> None:
        """Test with single token sequence."""
        batch, seqlen, nheads, headdim = 2, 1, 4, 64
        q, k, v = make_qkv(mlx, batch, seqlen, seqlen, nheads, nheads, headdim)

        out = flash_attn_mlx(q, k, v, causal=True)

        assert out.shape == (batch, 1, nheads, headdim)

    def test_single_head(self, mlx: Any, flash_attn_mlx: Callable[..., Any]) -> None:
        """Test with single attention head."""
        batch, seqlen, nheads, headdim = 2, 64, 1, 64
        q, k, v = make_qkv(mlx, batch, seqlen, seqlen, nheads, nheads, headdim)

        out = flash_attn_mlx(q, k, v, causal=True)

        assert out.shape == (batch, seqlen, 1, headdim)

    def test_invalid_shape(self, mlx: Any, flash_attn_mlx: Callable[..., Any]) -> None:
        """Test error handling for invalid shapes."""
        batch, seqlen, nheads, headdim = 2, 64, 4, 64

        # Create 3D tensor (invalid)
        q = mlx.random.normal((batch, seqlen, headdim)).astype(mlx.float16)
        k = mlx.random.normal((batch, seqlen, headdim)).astype(mlx.float16)
        v = mlx.random.normal((batch, seqlen, headdim)).astype(mlx.float16)

        with pytest.raises(ValueError, match="must be 4D"):
            flash_attn_mlx(q, k, v)

    def test_head_dim_mismatch(self, mlx: Any, flash_attn_mlx: Callable[..., Any]) -> None:
        """Test error handling for head dimension mismatch."""
        batch, seqlen, nheads = 2, 64, 4

        q = mlx.random.normal((batch, seqlen, nheads, 64)).astype(mlx.float16)
        k = mlx.random.normal((batch, seqlen, nheads, 128)).astype(mlx.float16)  # Different headdim
        v = mlx.random.normal((batch, seqlen, nheads, 128)).astype(mlx.float16)

        with pytest.raises(ValueError, match="Head dimensions must match"):
            flash_attn_mlx(q, k, v)


# ============================================================================
# Utils Tests
# ============================================================================

class TestUtils:
    """Test utility functions."""

    def test_check_args_valid(self, mlx: Any) -> None:
        """Test check_args with valid inputs."""
        from flash_attn.flash_attn_mlx.utils import check_args

        batch, seqlen, nheads, headdim = 2, 64, 4, 64
        q, k, v = make_qkv(mlx, batch, seqlen, seqlen, nheads, nheads, headdim)

        # Should not raise
        check_args(q, k, v)

    def test_check_args_gqa(self, mlx: Any) -> None:
        """Test check_args with GQA heads."""
        from flash_attn.flash_attn_mlx.utils import check_args

        batch, seqlen, headdim = 2, 64, 64
        q, k, v = make_qkv(mlx, batch, seqlen, seqlen, nheads=8, nheads_k=2, headdim=headdim)

        # Should not raise - 8 is divisible by 2
        check_args(q, k, v)

    def test_check_args_invalid_gqa(self, mlx: Any) -> None:
        """Test check_args rejects invalid GQA configuration."""
        from flash_attn.flash_attn_mlx.utils import check_args

        batch, seqlen, headdim = 2, 64, 64
        q, k, v = make_qkv(mlx, batch, seqlen, seqlen, nheads=7, nheads_k=3, headdim=headdim)

        with pytest.raises(ValueError, match="divisible"):
            check_args(q, k, v)


# ============================================================================
# Params Tests
# ============================================================================

class TestParams:
    """Test parameter dataclasses."""

    def test_attention_params_creation(self, mlx: Any) -> None:
        """Test creating AttentionParams."""
        from flash_attn.flash_attn_mlx.params import create_attention_params

        batch, seqlen, nheads, headdim = 2, 64, 4, 64
        q, k, v = make_qkv(mlx, batch, seqlen, seqlen, nheads, nheads, headdim)

        params = create_attention_params(q, k, v, causal=True)

        assert params.batch == batch
        assert params.seqlen_q == seqlen
        assert params.seqlen_k == seqlen
        assert params.nheads == nheads
        assert params.headdim == headdim
        assert params.causal is True
        assert params.softmax_scale == pytest.approx(1.0 / math.sqrt(headdim))

    def test_attention_params_custom_scale(self, mlx: Any) -> None:
        """Test AttentionParams with custom scale."""
        from flash_attn.flash_attn_mlx.params import create_attention_params

        batch, seqlen, nheads, headdim = 2, 64, 4, 64
        q, k, v = make_qkv(mlx, batch, seqlen, seqlen, nheads, nheads, headdim)

        custom_scale = 0.5
        params = create_attention_params(q, k, v, softmax_scale=custom_scale)

        assert params.softmax_scale == custom_scale


# ============================================================================
# Kernel Loader Tests
# ============================================================================

class TestKernelLoader:
    """Test kernel loading utilities."""

    def test_kernel_dir_exists(self) -> None:
        """Test kernel directory exists."""
        from flash_attn.flash_attn_mlx.kernels import get_kernel_dir

        kernel_dir = get_kernel_dir()
        assert kernel_dir.exists()

    def test_load_kernel_source(self) -> None:
        """Test loading kernel source files."""
        from flash_attn.flash_attn_mlx.kernels import load_kernel_source

        # Should be able to load forward kernel
        source = load_kernel_source("attention_fwd")
        assert len(source) > 0
        assert "flash_attention_forward" in source

    def test_load_all_kernels(self) -> None:
        """Test loading all kernels."""
        from flash_attn.flash_attn_mlx.kernels import load_all_kernels

        kernels = load_all_kernels()
        assert "attention_fwd" in kernels
        assert "attention_bwd_dq" in kernels
        assert "attention_bwd_dkv" in kernels
        assert "utils" in kernels


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

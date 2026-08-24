"""Correctness tests.

The acceptance bar for this repo: the hand-written attention must match
PyTorch's own fused reference to floating-point tolerance. Anything less
means the implementation is plausible rather than correct.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from attn.numpy_attention import MultiHeadAttention as NumpyMHA
from attn.numpy_attention import causal_mask as np_causal_mask
from attn.numpy_attention import softmax as np_softmax
from attn.torch_attention import MultiHeadAttention as TorchMHA

CONFIGS = [
    pytest.param(64, 8, 8, id="MHA"),
    pytest.param(64, 8, 4, id="GQA-2groups"),
    pytest.param(64, 8, 2, id="GQA-4groups"),
    pytest.param(64, 8, 1, id="MQA"),
]


def _inputs(b=2, t=7, d=64, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(b, t, d))
    return x, torch.from_numpy(x).double()


# --- softmax -----------------------------------------------------------


def test_softmax_matches_torch():
    x = np.random.default_rng(0).normal(size=(3, 5, 11)) * 10
    np.testing.assert_allclose(
        np_softmax(x), torch.from_numpy(x).softmax(-1).numpy(), rtol=1e-12, atol=1e-12
    )


def test_softmax_survives_extreme_logits():
    """The max-subtraction trick is load-bearing: exp(800) overflows."""
    x = np.array([[800.0, 799.0, -800.0]])
    out = np_softmax(x)
    assert np.isfinite(out).all(), "overflowed -- stability trick is broken"
    np.testing.assert_allclose(out.sum(), 1.0, rtol=1e-12)


# --- masking -----------------------------------------------------------


def test_causal_mask_square_is_lower_triangular():
    np.testing.assert_array_equal(np_causal_mask(4), np.tril(np.ones((4, 4), bool)))


def test_causal_mask_decode_row_sees_all_history():
    """T=1, S=n is the incremental-decode case: the one query row must see
    every key so far. A mask built without the s-t offset would expose only
    key 0 -- the bug is invisible when T==S and only bites during generation."""
    assert np_causal_mask(1, 5).all(), "decode query cannot see full history"


# --- equivalence -------------------------------------------------------


@pytest.mark.parametrize("d_model,n_heads,n_kv", CONFIGS)
def test_numpy_matches_torch(d_model, n_heads, n_kv):
    x_np, x_pt = _inputs(d=d_model)
    np_mha = NumpyMHA(d_model, n_heads, n_kv, seed=3)
    pt_mha = TorchMHA(d_model, n_heads, n_kv).double().load_numpy_weights(np_mha)

    out_np, w_np = np_mha.forward(x_np)
    with torch.no_grad():
        out_pt, w_pt = pt_mha(x_pt)

    np.testing.assert_allclose(out_np, out_pt.numpy(), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(w_np, w_pt.numpy(), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("d_model,n_heads,n_kv", CONFIGS)
def test_matches_pytorch_fused_reference(d_model, n_heads, n_kv):
    """Check against F.scaled_dot_product_attention -- an independent
    implementation we did not write. Passing our own two implementations
    against each other only proves they share a bug."""
    x_np, x_pt = _inputs(d=d_model)
    np_mha = NumpyMHA(d_model, n_heads, n_kv, seed=5)
    pt_mha = TorchMHA(d_model, n_heads, n_kv).double().load_numpy_weights(np_mha)

    b, t, _ = x_pt.shape
    with torch.no_grad():
        q = pt_mha._split(pt_mha.w_q(x_pt), n_heads)
        k = pt_mha._split(pt_mha.w_k(x_pt), n_kv)
        v = pt_mha._split(pt_mha.w_v(x_pt), n_kv)
        from attn.torch_attention import repeat_kv
        ref = F.scaled_dot_product_attention(
            q, repeat_kv(k, n_heads // n_kv), repeat_kv(v, n_heads // n_kv),
            is_causal=True,
        )
        ref = pt_mha.w_o(ref.transpose(1, 2).reshape(b, t, d_model))

    np.testing.assert_allclose(np_mha.forward(x_np)[0], ref.numpy(), rtol=1e-10, atol=1e-10)


# --- properties --------------------------------------------------------


def test_causality_future_tokens_get_zero_weight():
    x_np, _ = _inputs(t=6)
    _, w = NumpyMHA(64, 8, seed=1).forward(x_np, causal=True)
    for i in range(6):
        assert np.allclose(w[:, :, i, i + 1:], 0.0), f"query {i} leaked future keys"


def test_attention_weights_are_a_distribution():
    x_np, _ = _inputs()
    _, w = NumpyMHA(64, 8, seed=2).forward(x_np)
    np.testing.assert_allclose(w.sum(-1), 1.0, rtol=1e-12)
    assert (w >= 0).all()


def test_changing_a_future_token_cannot_change_the_present():
    """The strongest causality check: perturb the last position and every
    earlier output must be bit-identical."""
    x, _ = _inputs(t=6)
    mha = NumpyMHA(64, 8, seed=7)
    out_a, _ = mha.forward(x)
    x2 = x.copy()
    x2[:, -1, :] += 100.0
    out_b, _ = mha.forward(x2)
    np.testing.assert_allclose(out_a[:, :-1], out_b[:, :-1], rtol=1e-12, atol=1e-12)


# --- the arithmetic that motivates GQA ---------------------------------


def test_kv_cache_scales_with_kv_heads_not_query_heads():
    mha = NumpyMHA(64, 8, 8).kv_cache_bytes(1024)
    gqa = NumpyMHA(64, 8, 2).kv_cache_bytes(1024)
    mqa = NumpyMHA(64, 8, 1).kv_cache_bytes(1024)
    assert mha == 4 * gqa == 8 * mqa, "GQA/MQA saving is not being realised"


def test_kv_cache_formula_is_exact():
    """2 * n_kv_heads * d_head * seq * batch * dtype_bytes, per layer."""
    m = NumpyMHA(512, 8, 2)
    assert m.kv_cache_bytes(1024, batch=4, dtype_bytes=2) == 2 * 2 * 64 * 1024 * 4 * 2

"""Tests for the KV-cache arithmetic.

These lock down the formula itself. If a number in the README changes, one
of these should fail first.
"""

from __future__ import annotations

import pytest

from attn.kv_memory import GPUS, MODELS, ModelConfig, hypothetical_mha


def test_kv_per_token_formula():
    """2 * layers * kv_heads * d_head * dtype_bytes, hand-checked."""
    c = MODELS["llama-3-8b"]                       # 32 layers, 8 kv-heads, d_head 128
    assert c.kv_bytes_per_token(2) == 2 * 32 * 8 * 128 * 2 == 131_072


def test_kv_ignores_query_head_count():
    """The whole point of GQA: cache size depends on kv_heads only."""
    base = dict(n_layers=32, n_kv_heads=8, d_head=128, weight_gib=16.0, max_context=8192)
    few = ModelConfig("few-q", n_heads=8, **base)
    many = ModelConfig("many-q", n_heads=64, **base)
    assert few.kv_bytes_per_token() == many.kv_bytes_per_token()


@pytest.mark.parametrize("name", list(MODELS))
def test_gqa_saving_matches_group_size(name):
    """Switching to hypothetical MHA must inflate the cache by exactly the
    GQA group size -- no more, no less."""
    cfg = MODELS[name]
    mha = hypothetical_mha(cfg)
    ratio = mha.kv_bytes_per_token() / cfg.kv_bytes_per_token()
    assert ratio == pytest.approx(cfg.gqa_group_size)


def test_kv_scales_linearly_in_seq_and_batch():
    c = MODELS["llama-3-8b"]
    assert c.kv_gib(2048, 1) == pytest.approx(c.kv_gib(1024, 1) * 2)
    assert c.kv_gib(1024, 4) == pytest.approx(c.kv_gib(1024, 1) * 4)


def test_batch_ceiling_falls_as_context_grows():
    """Concurrency and context trade off against each other. This is why a
    deployment imposes a context cap: it is protecting throughput."""
    c = MODELS["llama-3-8b"]
    gpu = GPUS["H100-80G"]
    batches = [c.max_batch(n, gpu) for n in (2048, 4096, 8192, 16384)]
    assert batches == sorted(batches, reverse=True)
    assert all(b > 0 for b in batches)


def test_gqa_buys_real_concurrency():
    """The headline claim in the README: 4x the batch on an H100."""
    c = MODELS["llama-3-8b"]
    gpu = GPUS["H100-80G"]
    assert c.max_batch(8192, gpu) >= 4 * hypothetical_mha(c).max_batch(8192, gpu)


def test_kv_overtakes_weights_at_long_context():
    """Counter-intuitive and worth asserting: at long context and modest
    batch, the cache is larger than the model."""
    c = MODELS["llama-3-8b"]
    assert c.kv_vs_weights(8192, 1) < 1.0
    assert c.kv_vs_weights(32768, 8) > 1.0


def test_no_batch_fits_when_weights_exceed_gpu():
    """70B at fp16 does not fit on a 40GB card; the answer is 0, not a crash."""
    assert MODELS["llama-3-70b"].max_batch(8192, GPUS["A100-40G"]) == 0

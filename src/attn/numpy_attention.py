"""Scaled dot-product and multi-head attention, in NumPy, from first principles.

No framework attention primitives. The point is to be able to derive every
shape and every constant, not to call a fast kernel.

Shape convention used throughout:
    B  batch
    T  query positions      (target / current)
    S  key-value positions  (source; S == T for self-attention)
    H  number of query heads
    Hk number of key-value heads   (Hk == H for MHA, 1 for MQA, H/G for GQA)
    D  head dimension
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.floating]


def softmax(x: Array, axis: int = -1) -> Array:
    """Numerically stable softmax.

    Subtracting the row max before exponentiating changes nothing
    mathematically -- softmax is invariant to a constant shift -- but it
    keeps exp() away from overflow. Without it, a logit of ~800 overflows
    float64 and the whole row becomes NaN. This is the same trick that
    online-softmax (and therefore FlashAttention) generalises to a
    streaming setting.
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    q: Array,            # (B, H, T, D)
    k: Array,            # (B, H, S, D)
    v: Array,            # (B, H, S, D)
    mask: Array | None = None,   # (T, S) or broadcastable; True = keep
) -> tuple[Array, Array]:
    """Return (output, attention_weights).

    attn = softmax(QK^T / sqrt(D)) V

    Why the 1/sqrt(D)?
        q and k entries are roughly independent with unit variance, so their
        dot product over D dimensions has variance ~D and standard deviation
        ~sqrt(D). Feeding that into softmax unscaled means the logit spread
        grows with head dimension, softmax saturates towards one-hot, and the
        gradient through it vanishes. Dividing by sqrt(D) holds the logit
        variance at ~1 regardless of D, which keeps gradients alive.
        This is why it is sqrt(D) and not D or log(D): we are normalising a
        standard deviation, not a variance.
    """
    d = q.shape[-1]
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(d)   # (B, H, T, S)

    if mask is not None:
        # -inf (not a large negative number) so exp() gives exactly 0 and
        # masked positions contribute nothing to the softmax denominator.
        scores = np.where(mask, scores, -np.inf)

    weights = softmax(scores, axis=-1)                 # (B, H, T, S)
    return weights @ v, weights                        # (B, H, T, D)


def causal_mask(t: int, s: int | None = None) -> Array:
    """Lower-triangular keep-mask: query i may attend to keys 0..i.

    For incremental decode T=1 while S grows, so the single query row must
    be allowed to see every key so far -- which is what the k=s-t offset
    below produces. Getting this offset wrong is the classic KV-cache bug:
    it silently works at T=S and breaks only during generation.
    """
    if s is None:
        s = t
    return np.tril(np.ones((t, s), dtype=bool), k=s - t)


def repeat_kv(x: Array, n_rep: int) -> Array:
    """Expand Hk key/value heads to H query heads (GQA/MQA -> MHA shape).

    This is a broadcast for arithmetic purposes only. The memory saving of
    GQA is real precisely because the *cache* stores Hk heads; only the
    computation expands. Materialising this in the cache would throw away
    the entire benefit.
    """
    if n_rep == 1:
        return x
    b, hk, s, d = x.shape
    return np.broadcast_to(x[:, :, None], (b, hk, n_rep, s, d)).reshape(b, hk * n_rep, s, d)


class MultiHeadAttention:
    """MHA / MQA / GQA in one code path -- they differ only in n_kv_heads.

        n_kv_heads == n_heads  -> MHA   (one K/V per query head)
        n_kv_heads == 1        -> MQA   (all query heads share one K/V)
        1 < n_kv_heads < n_heads -> GQA (groups of query heads share a K/V)

    The progression MHA -> MQA -> GQA was driven by KV-cache bytes, not by
    FLOPs. See notes/why-gqa.md for the arithmetic.
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int | None = None,
                 seed: int = 0) -> None:
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        if n_heads % self.n_kv_heads:
            raise ValueError(
                f"n_heads {n_heads} not divisible by n_kv_heads {self.n_kv_heads}"
            )

        self.d_model, self.n_heads = d_model, n_heads
        self.d_head = d_model // n_heads
        self.n_rep = n_heads // self.n_kv_heads

        rng = np.random.default_rng(seed)
        kv_dim = self.n_kv_heads * self.d_head
        # Xavier-ish init; scale keeps activations O(1) so the sqrt(D)
        # argument above is actually exercised by the tests.
        s = 1.0 / np.sqrt(d_model)
        self.w_q = rng.normal(0, s, (d_model, d_model))
        self.w_k = rng.normal(0, s, (d_model, kv_dim))
        self.w_v = rng.normal(0, s, (d_model, kv_dim))
        self.w_o = rng.normal(0, s, (d_model, d_model))

    def _split(self, x: Array, n_heads: int) -> Array:
        """(B, T, n_heads*D) -> (B, n_heads, T, D)"""
        b, t, _ = x.shape
        return x.reshape(b, t, n_heads, self.d_head).transpose(0, 2, 1, 3)

    def forward(self, x: Array, causal: bool = True) -> tuple[Array, Array]:
        b, t, _ = x.shape

        q = self._split(x @ self.w_q, self.n_heads)      # (B, H,  T, D)
        k = self._split(x @ self.w_k, self.n_kv_heads)   # (B, Hk, T, D)
        v = self._split(x @ self.w_v, self.n_kv_heads)   # (B, Hk, T, D)

        k = repeat_kv(k, self.n_rep)                     # (B, H, T, D)
        v = repeat_kv(v, self.n_rep)

        mask = causal_mask(t) if causal else None
        out, weights = scaled_dot_product_attention(q, k, v, mask)

        # (B, H, T, D) -> (B, T, H*D)
        out = out.transpose(0, 2, 1, 3).reshape(b, t, self.d_model)
        return out @ self.w_o, weights

    def kv_cache_bytes(self, seq_len: int, batch: int = 1, dtype_bytes: int = 2) -> int:
        """Bytes held by the KV cache for this config.

            2 (K and V) * layers * n_kv_heads * d_head * seq_len * batch * dtype_bytes

        This method covers ONE layer. Note it scales with n_kv_heads, not
        n_heads -- that single fact is the whole reason GQA exists.
        """
        return 2 * self.n_kv_heads * self.d_head * seq_len * batch * dtype_bytes

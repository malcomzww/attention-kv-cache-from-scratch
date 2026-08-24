"""The same attention in PyTorch, weight-compatible with the NumPy version.

Deliberately does NOT use nn.MultiheadAttention or
F.scaled_dot_product_attention for the forward path -- the point is to own
every shape. F.sdpa is used only in the test suite, as an independent
reference to check against.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """(B, Hk, S, D) -> (B, Hk*n_rep, S, D). Expands for arithmetic only."""
    if n_rep == 1:
        return x
    b, hk, s, d = x.shape
    return x[:, :, None].expand(b, hk, n_rep, s, d).reshape(b, hk * n_rep, s, d)


def causal_mask(t: int, s: int, device: torch.device | None = None) -> Tensor:
    """Keep-mask where query i sees keys 0..i.

    The k=s-t diagonal offset matters for incremental decode: with T=1 and
    S=n, the single query row must see all n keys. A mask built with k=0
    would let it see only key 0 -- a bug that is invisible at T==S and only
    appears during generation.
    """
    return torch.ones(t, s, dtype=torch.bool, device=device).tril(diagonal=s - t)


class MultiHeadAttention(torch.nn.Module):
    """MHA / MQA / GQA, selected by n_kv_heads. Linear layers are bias-free
    to match the NumPy implementation exactly."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int | None = None) -> None:
        super().__init__()
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
        kv_dim = self.n_kv_heads * self.d_head

        self.w_q = torch.nn.Linear(d_model, d_model, bias=False)
        self.w_k = torch.nn.Linear(d_model, kv_dim, bias=False)
        self.w_v = torch.nn.Linear(d_model, kv_dim, bias=False)
        self.w_o = torch.nn.Linear(d_model, d_model, bias=False)

    def _split(self, x: Tensor, n_heads: int) -> Tensor:
        b, t, _ = x.shape
        return x.view(b, t, n_heads, self.d_head).transpose(1, 2)

    def forward(self, x: Tensor, causal: bool = True) -> tuple[Tensor, Tensor]:
        b, t, _ = x.shape
        q = self._split(self.w_q(x), self.n_heads)
        k = repeat_kv(self._split(self.w_k(x), self.n_kv_heads), self.n_rep)
        v = repeat_kv(self._split(self.w_v(x), self.n_kv_heads), self.n_rep)

        scores = q @ k.transpose(-1, -2) / math.sqrt(self.d_head)
        if causal:
            scores = scores.masked_fill(~causal_mask(t, t, x.device), float("-inf"))
        weights = scores.softmax(dim=-1)

        out = (weights @ v).transpose(1, 2).reshape(b, t, self.d_model)
        return self.w_o(out), weights

    def kv_cache_bytes(self, seq_len: int, batch: int = 1, dtype_bytes: int = 2) -> int:
        """Per-layer KV cache bytes. Scales with n_kv_heads, not n_heads."""
        return 2 * self.n_kv_heads * self.d_head * seq_len * batch * dtype_bytes

    @torch.no_grad()
    def load_numpy_weights(self, np_mha) -> MultiHeadAttention:
        """Copy weights from the NumPy module so the two can be compared.

        nn.Linear stores weight as (out_features, in_features) and computes
        x @ W.T, whereas the NumPy version computes x @ W with W as
        (in, out). Hence the transpose.
        """
        for dst, src in (
            (self.w_q, np_mha.w_q), (self.w_k, np_mha.w_k),
            (self.w_v, np_mha.w_v), (self.w_o, np_mha.w_o),
        ):
            dst.weight.copy_(torch.from_numpy(src.T.copy()).to(dst.weight.dtype))
        return self

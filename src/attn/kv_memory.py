"""KV-cache memory arithmetic, and what it implies for batch size.

This is the module behind the claim that KV cache -- not model weights --
is what limits concurrency at long context. Every number here is computed,
not quoted.
"""

from __future__ import annotations

from dataclasses import dataclass

GIB = 1024 ** 3


@dataclass(frozen=True)
class ModelConfig:
    """Shape parameters that determine KV-cache size."""

    name: str
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_head: int
    weight_gib: float          # weights at the serving dtype
    max_context: int

    @property
    def d_model(self) -> int:
        return self.n_heads * self.d_head

    @property
    def gqa_group_size(self) -> int:
        return self.n_heads // self.n_kv_heads

    def kv_bytes_per_token(self, dtype_bytes: int = 2) -> int:
        """Bytes of KV cache added by ONE token, across all layers.

            2 (K and V) * n_layers * n_kv_heads * d_head * dtype_bytes

        Note what is absent: n_heads. The cache scales with the number of
        *key/value* heads, which is the entire reason GQA exists.
        """
        return 2 * self.n_layers * self.n_kv_heads * self.d_head * dtype_bytes

    def kv_gib(self, seq_len: int, batch: int = 1, dtype_bytes: int = 2) -> float:
        return self.kv_bytes_per_token(dtype_bytes) * seq_len * batch / GIB

    def max_batch(self, seq_len: int, gpu_gib: float, dtype_bytes: int = 2,
                  overhead_gib: float = 2.0) -> int:
        """How many concurrent sequences fit at this context length.

        GPU memory is spent on three things: weights (fixed), activations
        and framework overhead (roughly fixed), and KV cache (grows with
        batch * seq_len). Only the third scales, so it is what sets the
        concurrency ceiling -- and therefore throughput, and therefore
        cost per token.
        """
        free = gpu_gib - self.weight_gib - overhead_gib
        if free <= 0:
            return 0
        per_seq = self.kv_gib(seq_len, 1, dtype_bytes)
        return int(free / per_seq) if per_seq else 0

    def kv_vs_weights(self, seq_len: int, batch: int, dtype_bytes: int = 2) -> float:
        """KV cache as a multiple of weight memory. Crosses 1.0 sooner than
        most people expect, which is the surprising part."""
        return self.kv_gib(seq_len, batch, dtype_bytes) / self.weight_gib


# Public config values, from published model cards.
MODELS = {
    "llama-3-8b": ModelConfig("Llama-3-8B", 32, 32, 8, 128, 16.0, 8192),
    "llama-3-70b": ModelConfig("Llama-3-70B", 80, 64, 8, 128, 140.0, 8192),
    "qwen2.5-7b": ModelConfig("Qwen2.5-7B", 28, 28, 4, 128, 15.2, 32768),
    "mistral-7b": ModelConfig("Mistral-7B", 32, 32, 8, 128, 14.5, 32768),
}

GPUS = {"A100-40G": 40.0, "A100-80G": 80.0, "H100-80G": 80.0, "H200-141G": 141.0}


def hypothetical_mha(cfg: ModelConfig) -> ModelConfig:
    """The same model as if it had never adopted GQA -- n_kv_heads == n_heads.

    Used to quantify what GQA actually bought, rather than asserting it.
    """
    return ModelConfig(
        f"{cfg.name} (MHA)", cfg.n_layers, cfg.n_heads, cfg.n_heads,
        cfg.d_head, cfg.weight_gib, cfg.max_context,
    )

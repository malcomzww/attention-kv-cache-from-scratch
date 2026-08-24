# Attention and the KV cache, from first principles

Scaled dot-product attention, MHA/MQA/GQA, and the KV cache implemented by
hand in NumPy and PyTorch — with the memory arithmetic that explains why
the architecture moved the way it did.

No `nn.MultiheadAttention`, no `F.scaled_dot_product_attention` in the
forward path. Those appear only in the test suite, as an independent
reference to check against.

```
$ python -m pytest -q
28 passed in 1.60s

$ python scripts/generate_tables.py
wrote results/kv-analysis.md
```

```python
>>> from attn.kv_memory import MODELS, GPUS, hypothetical_mha
>>> c = MODELS["llama-3-8b"]
>>> c.kv_bytes_per_token()          # 2 * 32 layers * 8 kv-heads * 128 * 2 bytes
131072
>>> c.max_batch(8192, GPUS["H100-80G"])          # with GQA
62
>>> hypothetical_mha(c).max_batch(8192, GPUS["H100-80G"])   # had it been MHA
15
```

**That last pair is the whole repo in two lines.** Same model, same GPU,
same context length. Grouped-query attention is the difference between 15
and 62 concurrent sequences — a 4× throughput gap, and therefore a 4× cost
gap, from one architectural choice.

## Why this exists

Attention is usually explained as a formula and then handed to a library.
That leaves two things unexplained that matter in production:

1. **Why the progression MHA → MQA → GQA happened at all.** It was not
   about FLOPs. It was about KV-cache bytes, and the arithmetic is simple
   enough to do by hand — so this repo does it.
2. **Why a serving deployment caps context.** A context limit looks like a
   correctness constraint and is actually a throughput decision. Doubling
   context halves concurrency.

I hit the second one in production: a ~9.4k-token document rejected by an
8k deployment cap. The cap was not arbitrary, and understanding *why*
changes which of the four available fixes you reach for.

## Quickstart

```bash
git clone https://github.com/moclamzw/attention-kv-cache-from-scratch
cd attention-kv-cache-from-scratch
uv sync --extra dev          # or: pip install -e ".[dev]"
python -m pytest -q
python scripts/generate_tables.py
```

## Results

Full generated tables: [`results/kv-analysis.md`](results/kv-analysis.md).

### Cache growth per token

`2 × n_layers × n_kv_heads × d_head × dtype_bytes`

| model | layers | q-heads | kv-heads | group | KV/token | KV @ 8k |
|---|---|---|---|---|---|---|
| Llama-3-8B | 32 | 32 | 8 | 4× | 131,072 B | 1.00 GiB |
| Llama-3-70B | 80 | 64 | 8 | 8× | 327,680 B | 2.50 GiB |
| Qwen2.5-7B | 28 | 28 | 4 | 7× | 57,344 B | 0.44 GiB |

Note what is **absent** from that formula: `n_heads`. The cache scales with
the number of key/value heads, not query heads. That single asymmetry is
the entire reason GQA exists.

### Concurrency ceiling — Llama-3-8B

| context | A100-40G | A100-80G | H100-80G | H200-141G |
|---|---|---|---|---|
| 2,048 | 88 | 248 | 248 | 492 |
| 4,096 | 44 | 124 | 124 | 246 |
| 8,192 | 22 | 62 | 62 | 123 |
| 16,384 | 11 | 31 | 31 | 61 |
| 32,768 | 5 | 15 | 15 | 30 |

Assumes 16 GiB weights, 2 GiB activation/framework overhead, fp16 KV.

### When the cache overtakes the model

KV cache as a multiple of weight memory:

| context | batch 1 | batch 8 | batch 32 |
|---|---|---|---|
| 8,192 | 0.06× | 0.50× | 2.00× |
| 32,768 | 0.25× | 2.00× | 8.00× |
| 131,072 | 1.00× | 8.00× | 32.00× |

A capacity plan that budgets only for weights is wrong by more than an
order of magnitude at long context.

## Design decisions

**One code path for MHA, MQA and GQA.** They differ only in `n_kv_heads`.
Writing them as three classes would have hidden the fact that they are one
architecture with one parameter.

**`repeat_kv` expands for arithmetic only.** The cache stores `n_kv_heads`.
Materialising the expansion in the cache would throw away the entire memory
saving — an easy and silent mistake.

**Tested against an implementation I did not write.** The NumPy and PyTorch
paths are weight-compatible, but checking them only against each other
would prove they share a bug. Both are checked against
`F.scaled_dot_product_attention`.

**The causal mask carries an `s - t` offset.** During incremental decode
`T=1` while `S` grows, so the single query row must see all history. A mask
built with `k=0` exposes only key 0 — invisible while `T == S`, and it
appears only during generation. It has its own test.

**Every README number is generated.** `scripts/generate_tables.py` rebuilds
them from the code, and the formula is locked by tests, so a changed number
fails a test before it reaches this file.

## Limitations

- **This is a reference implementation, not a fast one.** No FlashAttention,
  no kernel fusion, no CUDA. It materialises the full `T × S` attention
  matrix, which is exactly what FlashAttention avoids. Correctness and
  legibility over speed, deliberately.
- **The memory model counts KV cache, weights, and a flat overhead term.**
  It ignores fragmentation, which is real — PagedAttention exists because
  naive contiguous allocation wastes a large fraction of the cache. Treat
  the concurrency numbers as an upper bound.
- **Model configs are from public model cards**, not measured on hardware I
  own. The formula is exact; the inputs are as accurate as those cards.
- No RoPE, no attention sinks, no sliding-window attention yet.

## Repository layout

```
src/attn/numpy_attention.py   first-principles NumPy implementation
src/attn/torch_attention.py   weight-compatible PyTorch version
src/attn/kv_memory.py         cache arithmetic and batch-size ceilings
tests/                        28 tests: equivalence, causality, arithmetic
scripts/generate_tables.py    regenerates every table in this README
results/kv-analysis.md        generated output
```

## License

MIT

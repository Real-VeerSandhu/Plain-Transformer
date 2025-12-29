Development plan (CPU-only, no real weights)
Goal of the project

Build a decoder-only transformer that can run in “real time” on CPU for small configs, and that visibly demonstrates KV-cache vs no-cache with clean profiling + scaling sweeps.

Milestones (in order)
Milestone 1 — “It runs” (forward pass + sampling)

Outcome: python run.py --prompt "hello" --steps 50 streams tokens.

Build

Char-level tokenizer (256 vocab)

Random initialized weights (seeded for repeatability)

Decoder-only transformer:

Embedding

RMSNorm

Multi-head self-attention (causal mask)

MLP (GELU or SwiGLU)

Final norm + LM head

Greedy sampling (argmax)

Acceptance checks

Shapes always consistent: [B,T,d_model]

Deterministic output given same seed + prompt

CPU-friendly default config

L=2, d_model=128, H=4, d_ff=256, max_seq=256

Milestone 2 — Correct causal attention + RoPE (still no KV)

Outcome: attention is correct and you can extend context without learned position embeddings.

Build

Implement RoPE applied to Q and K (per head)

Implement causal masking efficiently (no giant [T,T] boolean per step if possible for decode)

Acceptance checks

RoPE positions increase correctly during generation

No look-ahead: token t never attends to >t

Milestone 3 — Add KV cache (the star feature)

Outcome: Toggle --kv on/off and show a clear speed difference.

Build

KVCache per layer storing:

K_cache: [B,H,T,Dh]

V_cache: [B,H,T,Dh]

Two execution paths:

No cache: each decode step recomputes K,V for all tokens (slow)

Cache: prefill stores full K,V; decode appends only 1 token’s K,V

Acceptance checks

Equivalence test: for the same prompt, the logits for step t match between cache and no-cache (within small float tolerance)

Cache length grows exactly by 1 each decode token

Milestone 4 — Instrumentation + “real-time” streaming UX

Outcome: Your demo looks like a tiny serving engine.

Build

Separate prefill and decode timings

Per-token decode latency printed live:

ms/token

tokens/sec (rolling average)

Compute + memory estimates:

KV bytes = L * 2 * B * H * T * Dh * bytes_per_elem

Attention FLOPs rough estimate (optional)

Acceptance checks

Nice structured output (table-ish)

--verbose prints per-layer timings (optional)

Milestone 5 — Benchmark suite (the portfolio graph generator)

Outcome: bench.py produces convincing scaling results on CPU.

Build

Sweep prompt length T = [16, 32, 64, 128, 256, 512]

Sweep model sizes (small/medium):

Small: L2 d128 H4

Medium: L4 d256 H8

(Keep it CPU-viable)

Run each setting:

prefill time

average decode time over N tokens

KV on/off

Acceptance checks

Results show:

Prefill grows ~quadratically with T

Decode without cache grows with T (each step slower as context grows)

Decode with cache is ~stable per token (grows much slower)

Milestone 6 — Optimization pass (still pure CPU, still no weights)

Outcome: It feels snappy for small configs and the benchmark is stable.

Optimize (pick in this order)

Use float32 by default; optionally support float16 but CPU fp16 may be slower depending on backend.

Pre-allocate KV cache arrays for max_seq to avoid repeated concatenation.

Use incremental decode path that avoids building full masks each step.

Reduce Python overhead:

keep tensor ops in vectorized form

minimal per-token Python work

Acceptance checks

Medium config can stream at a readable pace on CPU

Benchmark runs without huge variance
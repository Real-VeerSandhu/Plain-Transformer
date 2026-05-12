# Plain Transformer

Plain Transformer is a decoder-only transformer language model implemented entirely from first principles in PyTorch. Every core component is hand-built, including causal self-attention, Rotary Positional Embeddings (RoPE), RMSNorm, SwiGLU feed-forward networks, KV caching, and a complete end-to-end training loop. This project exposes the underlying mechanics of modern language models so you can inspect, modify, and experiment with every part of the architecture.

Train a character-level language model on real text and watch loss and perplexity converge, visualize attention maps to see what each head attends to at every layer, inspect RoPE frequency patterns and causal masking behavior, and benchmark KV caching against naive decoding. The codebase is designed for experimentation, making it easy to swap SwiGLU for GELU, adjust model depth and width, change the number of attention heads, and study how these architectural choices affect model behavior.

---

## Quick start

```bash
pip install -r requirements.txt

# Generate with a random model (untrained)
python run.py

# Train on TinyShakespeare (auto-downloads)
python train.py --steps 3000

# Resume from checkpoint
python train.py --resume checkpoints/ckpt_final.pt --steps 6000

# Generate from trained model
python run.py --ckpt checkpoints/ckpt_final.pt --prompt "HAMLET:" --steps 200
```

---

## Architecture

### Attention (`transformer/attention.py`)

Causal multi-head self-attention with:
- **RoPE** on Q and K: rotates pairs of dimensions by position-dependent angles, encoding relative distance rather than absolute position
- **KV cache**: during decoding, stored K/V projections from prior steps are reused — O(1) per step instead of O(T²)
- **Weight capture**: pass `return_attn_weights=True` to collect `(B, H, T, T)` attention maps per layer

```
scores = (Q·Kᵀ) / √head_dim
attn   = softmax(scores + causal_mask)
out    = attn · V
```

### Positional Encoding (`transformer/rope.py`)

RoPE encodes position by rotating Q and K in 2D subspaces:
```
[x₀, x₁] → [x₀·cos(θ) - x₁·sin(θ),  x₀·sin(θ) + x₁·cos(θ)]
```
where θ = position × (1/10000^(2i/d)). Lower dimension indices rotate fast (short-range), higher indices rotate slow (long-range). No learned position parameters.

### Normalization

**RMSNorm** (not LayerNorm): normalizes by RMS only, skipping mean-centering. Faster and empirically equivalent.

### Feed-forward

**SwiGLU** (default): `W₃(SiLU(W₁x) ⊙ W₂x)`.
**GELU** (fallback): standard two-layer MLP.

---

## Training

```bash
python train.py \
  --steps 5000 \
  --d_model 256 --num_layers 6 --num_heads 8 --d_ff 1024 \
  --batch_size 64 --block_size 256 \
  --lr 3e-4 --warmup_steps 200
```

**LR schedule**: linear warmup → cosine decay to `--min_lr`.  
**Optimizer**: AdamW (β₁=0.9, β₂=0.95, weight_decay=0.1).  
**Gradient clipping**: `--grad_clip 1.0`.  
**Checkpoints**: saved every `--save_interval` steps to `checkpoints/`.

Logged per step: loss, perplexity, LR, gradient norm, throughput.

---

## Lab (`lab.py`)
Study internals & play with parameters.

```bash
# Causal mask structure
python lab.py --mode mask --seq_len 16

# RoPE frequency grid (low dim = high freq = position sensitive)
python lab.py --mode rope

# Attention heatmaps: what each token attends to
python lab.py --mode attn --prompt "To be or not to be"
python lab.py --mode attn --ckpt checkpoints/ckpt_final.pt --prompt "HAMLET:"

# Training dynamics: loss curve, perplexity
python lab.py --mode curves --ckpt checkpoints/ckpt_final.pt

# Architecture summary: param counts, attention math
python lab.py --mode summary
python lab.py --mode summary --d_model 256 --num_layers 6
```

All plots are saved to `demo-plots/` by default. Override with `--out-dir path/` or `--out file.png`.

---

## KV cache benchmark

```bash
python compare.py                         # default 201-char prompt, 40 steps
python compare.py --steps 100             # longer decode == bigger speedup
```

The speedup scales with prompt length because without caching each decode step reprocesses the full growing context (O(T²) attention) vs O(T) with cached K/V.

---

## Model configuration

| Param | Default | Notes |
|---|---|---|
| `d_model` | 128 | Embedding / hidden dimension |
| `num_layers` | 4 | Transformer blocks |
| `num_heads` | 4 | Attention heads (head_dim = d_model/num_heads) |
| `d_ff` | 512 | FFN hidden dimension |
| `max_seq_len` | 512 | Maximum context length |
| `use_swiglu` | True | SwiGLU vs GELU FFN |
| `dropout` | 0.1 | Applied in attn and FFN (0.0 for inference) |

#!/usr/bin/env python3
"""
Transformer Internals Inspector — the lab's main exploration tool.

Usage:
    # Visualize attention patterns (randomly init'd model)
    python lab.py --mode attn --prompt "To be or not to be"

    # Load a trained checkpoint and inspect it
    python lab.py --mode attn --ckpt checkpoints/ckpt_final.pt --prompt "HAMLET:"

    # Show RoPE frequency patterns
    python lab.py --mode rope

    # Draw the causal attention mask
    python lab.py --mode mask --seq_len 16

    # Plot training curves from a checkpoint
    python lab.py --mode curves --ckpt checkpoints/ckpt_final.pt

    # Full architecture summary
    python lab.py --mode summary

Studying model internals:
    - Attention maps show WHAT each token attends to
    - RoPE frequencies show HOW position is encoded in each head dimension
    - Causal mask shows WHY autoregressive models can't peek forward
    - Training curves reveal dynamics: underfitting, overfitting, instability
"""
import argparse
import math
from pathlib import Path

import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not found. Install it for plots: pip install matplotlib")


from transformer import CharTokenizer, ModelConfig, Transformer
from transformer.rope import build_rope_cache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_mpl():
    if not HAS_MPL:
        raise SystemExit("This mode requires matplotlib. Run: pip install matplotlib")


def load_model(ckpt_path: str | None, cfg_kwargs: dict, device: torch.device):
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device)
        cfg: ModelConfig = ckpt["cfg"]
        model = Transformer(cfg).to(device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded checkpoint: {ckpt_path}")
        return model, ckpt
    cfg = ModelConfig(**cfg_kwargs)
    model = Transformer(cfg).to(device)
    print("Using randomly initialized model (no checkpoint).")
    return model, None


# ---------------------------------------------------------------------------
# Mode: attention heatmaps
# ---------------------------------------------------------------------------

def mode_attn(args, device):
    _require_mpl()
    model, _ = load_model(args.ckpt, _cfg_kwargs(args), device)
    model.eval()
    tokenizer = CharTokenizer()

    ids = torch.tensor([tokenizer.encode(args.prompt)], device=device)
    tokens = list(args.prompt)
    T = len(tokens)

    with torch.no_grad():
        _, _, attn_weights_list = model(ids, return_attn_weights=True)

    # attn_weights_list: list of (B, H, T_q, T_k) or None
    num_layers = len(attn_weights_list)
    num_heads = model.cfg.num_heads

    fig, axes = plt.subplots(num_layers, num_heads,
                             figsize=(num_heads * 2.5, num_layers * 2.5))
    if num_layers == 1:
        axes = [axes]
    if num_heads == 1:
        axes = [[ax] for ax in axes]

    fig.suptitle(f'Attention Maps | prompt: "{args.prompt}"', fontsize=12)

    for layer_idx, weights in enumerate(attn_weights_list):
        if weights is None:
            continue
        w = weights[0].cpu()  # (H, T_q, T_k)
        for head_idx in range(num_heads):
            ax = axes[layer_idx][head_idx]
            im = ax.imshow(w[head_idx].numpy(), vmin=0, vmax=1, cmap="Blues", aspect="auto")
            ax.set_title(f"L{layer_idx} H{head_idx}", fontsize=8)
            ax.set_xticks(range(T))
            ax.set_yticks(range(T))
            ax.set_xticklabels([repr(c)[1:-1] for c in tokens], fontsize=6, rotation=90)
            ax.set_yticklabels([repr(c)[1:-1] for c in tokens], fontsize=6)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    out = _resolve_out(args, "attention_maps.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Mode: RoPE frequency visualization
# ---------------------------------------------------------------------------

def mode_rope(args, device):
    _require_mpl()
    model, _ = load_model(args.ckpt, _cfg_kwargs(args), device)
    head_dim = model.cfg.head_dim
    seq_len = args.seq_len or 64

    cos_table, sin_table = build_rope_cache(seq_len, head_dim, device=torch.device("cpu"))
    # cos_table: (seq_len, head_dim/2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im0 = axes[0].imshow(cos_table.T.numpy(), aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
    axes[0].set_xlabel("Position")
    axes[0].set_ylabel("Frequency dimension")
    axes[0].set_title("RoPE cos(θ) — position × frequency grid")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(sin_table.T.numpy(), aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
    axes[1].set_xlabel("Position")
    axes[1].set_ylabel("Frequency dimension")
    axes[1].set_title("RoPE sin(θ) — position × frequency grid")
    plt.colorbar(im1, ax=axes[1])

    fig.suptitle(
        f"RoPE Frequencies  |  head_dim={head_dim}  |  seq_len={seq_len}\n"
        "Low dim indices = high-freq (position sensitive)  |  High dim = low-freq (long-range)",
        fontsize=10,
    )
    plt.tight_layout()
    out = _resolve_out(args, "rope_frequencies.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Mode: causal mask
# ---------------------------------------------------------------------------

def mode_mask(args, _device):
    _require_mpl()
    T = args.seq_len or 16
    mask = torch.tril(torch.ones(T, T)).numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(mask, cmap="Greens", vmin=0, vmax=1)
    ax.set_title(f"Causal (lower-triangular) attention mask  [seq_len={T}]")
    ax.set_xlabel("Key position (attend TO)")
    ax.set_ylabel("Query position (attend FROM)")

    # Annotate cells
    if T <= 12:
        for i in range(T):
            for j in range(T):
                ax.text(j, i, "1" if mask[i, j] else "×", ha="center", va="center",
                        fontsize=8, color="black" if mask[i, j] else "gray")

    ax.set_xticks(range(T))
    ax.set_yticks(range(T))
    plt.tight_layout()
    out = _resolve_out(args, "causal_mask.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Mode: training curves
# ---------------------------------------------------------------------------

def mode_curves(args, device):
    _require_mpl()
    if not args.ckpt:
        raise SystemExit("--ckpt required for curves mode")

    ckpt = torch.load(args.ckpt, map_location=device)
    train_losses = ckpt.get("train_losses", [])
    val_losses = ckpt.get("val_losses", [])

    if not train_losses:
        raise SystemExit("Checkpoint has no recorded training losses.")

    train_steps, train_vals = zip(*train_losses)
    train_ppls = [math.exp(min(v, 20)) for v in train_vals]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_steps, train_vals, label="train loss", alpha=0.8, linewidth=1.5)
    if val_losses:
        val_steps, val_vals = zip(*val_losses)
        ax1.plot(val_steps, val_vals, "o-", label="val loss", linewidth=2)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Cross-entropy loss")
    ax1.set_title("Training Dynamics — Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(train_steps, train_ppls, label="train ppl", alpha=0.8, linewidth=1.5)
    if val_losses:
        val_ppls = [math.exp(min(v, 20)) for v in val_vals]
        ax2.plot(val_steps, val_ppls, "o-", label="val ppl", linewidth=2)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Perplexity")
    ax2.set_title("Training Dynamics — Perplexity")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    final_step = ckpt.get("step", "?")
    cfg = ckpt.get("cfg")
    title = f"Training curves | step {final_step}"
    if cfg:
        title += f" | L={cfg.num_layers} d={cfg.d_model} H={cfg.num_heads}"
    fig.suptitle(title)

    plt.tight_layout()
    out = _resolve_out(args, "training_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Mode: architecture summary
# ---------------------------------------------------------------------------

def mode_summary(args, device):
    model, _ = load_model(args.ckpt, _cfg_kwargs(args), device)
    cfg = model.cfg

    total = sum(p.numel() for p in model.parameters())
    embed = model.token_embedding.weight.numel()
    attn_per_layer = sum(p.numel() for p in model.blocks[0].attn.parameters())
    ffn_per_layer = sum(p.numel() for p in model.blocks[0].ffn.parameters())
    norm_per_layer = sum(p.numel() for p in model.blocks[0].ln1.parameters()) * 2

    print("\n" + "=" * 60)
    print("TRANSFORMER ARCHITECTURE SUMMARY")
    print("=" * 60)
    print(f"  vocab_size:    {cfg.vocab_size}")
    print(f"  d_model:       {cfg.d_model}")
    print(f"  num_layers:    {cfg.num_layers}")
    print(f"  num_heads:     {cfg.num_heads}")
    print(f"  head_dim:      {cfg.head_dim}")
    print(f"  d_ff:          {cfg.d_ff}")
    print(f"  max_seq_len:   {cfg.max_seq_len}")
    print(f"  FFN type:      {'SwiGLU' if cfg.use_swiglu else 'GELU'}")
    print(f"  dropout:       {cfg.dropout}")
    print()
    print(f"  Parameters:")
    print(f"    embedding:          {embed:>10,}")
    print(f"    attn per layer:     {attn_per_layer:>10,}  × {cfg.num_layers} = {attn_per_layer * cfg.num_layers:,}")
    print(f"    ffn  per layer:     {ffn_per_layer:>10,}  × {cfg.num_layers} = {ffn_per_layer * cfg.num_layers:,}")
    print(f"    norms per layer:    {norm_per_layer:>10,}  × {cfg.num_layers} = {norm_per_layer * cfg.num_layers:,}")
    print(f"    TOTAL:              {total:>10,}")
    print()
    print(f"  Attention math:")
    print(f"    Q/K/V proj:  ({cfg.d_model}, {cfg.d_model}) each")
    print(f"    scores:      (B, {cfg.num_heads}, T, T)  scale=1/sqrt({cfg.head_dim})={1/math.sqrt(cfg.head_dim):.4f}")
    print(f"    KV cache:    2 × B × {cfg.num_layers} × T × {cfg.d_model} floats per step")
    print()
    print(f"  RoPE:")
    print(f"    base frequency: 10000")
    print(f"    dimensions: {cfg.head_dim // 2} rotation pairs per head")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_out(args, default_filename: str) -> Path:
    """Return the output path, creating the output directory if needed."""
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / default_filename


def _cfg_kwargs(args) -> dict:
    return dict(
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        use_swiglu=args.swiglu,
    )


def main():
    p = argparse.ArgumentParser(description="Transformer Internals Inspector",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--mode", choices=["attn", "rope", "mask", "curves", "summary"],
                   default="summary", help="What to visualize")
    p.add_argument("--ckpt", type=str, default=None, help="Checkpoint path")
    p.add_argument("--prompt", type=str, default="To be or not to be", help="Prompt for attn mode")
    p.add_argument("--seq_len", type=int, default=None, help="Sequence length for mask/rope modes")
    p.add_argument("--out", type=str, default=None, help="Exact output path (overrides --out-dir)")
    p.add_argument("--out-dir", dest="out_dir", type=str, default="demo-plots",
                   help="Directory for output plots (created if missing)")

    # Model config (used when no checkpoint)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--d_ff", type=int, default=512)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--swiglu", action="store_true", default=True)

    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dispatch = {
        "attn": mode_attn,
        "rope": mode_rope,
        "mask": mode_mask,
        "curves": mode_curves,
        "summary": mode_summary,
    }
    dispatch[args.mode](args, device)


if __name__ == "__main__":
    main()

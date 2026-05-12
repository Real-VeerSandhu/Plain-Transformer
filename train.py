#!/usr/bin/env python3
"""
Training loop for the from-scratch transformer.

Usage:
    python train.py                          # train on TinyShakespeare
    python train.py --data path/to/text.txt  # train on custom text
    python train.py --d_model 256 --num_layers 6 --steps 5000
    python train.py --resume checkpoints/ckpt_2000.pt

Training dynamics you can study:
    - Loss curve shape (rapid early drop, slower refinement)
    - Effect of learning rate on convergence speed
    - Gradient norms (stability indicator)
    - Perplexity vs model size trade-offs
"""
import argparse
import math
import os
import time
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.utils as nn_utils

from transformer import CharTokenizer, ModelConfig, Transformer

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
CHECKPOINT_DIR = Path("checkpoints")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_or_download_text(path: Optional[str]) -> str:
    if path and os.path.exists(path):
        return Path(path).read_text(encoding="utf-8")

    cache = Path(".data/tinyshakespeare.txt")
    if cache.exists():
        return cache.read_text(encoding="utf-8")

    print(f"Downloading TinyShakespeare from {SHAKESPEARE_URL} ...")
    cache.parent.mkdir(exist_ok=True)
    try:
        urllib.request.urlretrieve(SHAKESPEARE_URL, cache)
        return cache.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Download failed ({e}). Using a tiny synthetic corpus.")
        return (
            "To be or not to be that is the question whether tis nobler in the mind "
            "to suffer the slings and arrows of outrageous fortune or to take arms "
            "against a sea of troubles and by opposing end them. " * 200
        )


def make_dataset(text: str, tokenizer: CharTokenizer, val_frac: float = 0.1):
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split = int(len(ids) * (1 - val_frac))
    return ids[:split], ids[split:]


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: torch.device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix]).to(device)
    return x, y


# ---------------------------------------------------------------------------
# LR schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: Transformer, data: torch.Tensor, block_size: int,
             batch_size: int, device: torch.device, num_batches: int = 20) -> float:
    model.eval()
    total_loss = 0.0
    for _ in range(num_batches):
        x, y = get_batch(data, block_size, batch_size, device)
        _, loss, _ = model(x, targets=y)
        total_loss += loss.item()
    model.train()
    return total_loss / num_batches


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(model: Transformer, optimizer: torch.optim.Optimizer,
                    step: int, train_losses: list, val_losses: list, path: Path):
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    torch.save({
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "cfg": model.cfg,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }, path)
    print(f"  checkpoint saved -> {path}")


def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    cfg: ModelConfig = ckpt["cfg"]
    model = Transformer(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # Data
    text = load_or_download_text(args.data)
    tokenizer = CharTokenizer()
    train_data, val_data = make_dataset(text, tokenizer)
    print(f"Dataset: {len(text):,} chars | train {len(train_data):,} tokens | val {len(val_data):,} tokens")

    block_size = min(args.block_size, args.max_seq_len)

    # Model
    if args.resume:
        model, ckpt = load_checkpoint(args.resume, device)
        start_step = ckpt["step"] + 1
        train_losses = ckpt["train_losses"]
        val_losses = ckpt["val_losses"]
        print(f"Resumed from {args.resume} at step {ckpt['step']}")
    else:
        cfg = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            max_seq_len=args.max_seq_len,
            use_swiglu=args.swiglu,
            dropout=args.dropout,
        )
        model = Transformer(cfg).to(device)
        start_step = 0
        train_losses = []
        val_losses = []

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )
    if args.resume and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    # Training
    print(f"\nTraining for {args.steps} steps | batch={args.batch_size} block={block_size}")
    print(f"LR: {args.lr} (warmup {args.warmup_steps} steps, min {args.min_lr})\n")

    model.train()
    t0 = time.time()

    for step in range(start_step, args.steps):
        lr = get_lr(step, args.warmup_steps, args.steps, args.lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x, y = get_batch(train_data, block_size, args.batch_size, device)
        _, loss, _ = model(x, targets=y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn_utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        train_losses.append((step, loss.item()))

        # Logging
        if step % args.log_interval == 0:
            dt = time.time() - t0
            tok_per_sec = args.log_interval * args.batch_size * block_size / max(dt, 1e-6)
            print(f"step {step:>6} | loss {loss.item():.4f} | ppl {math.exp(loss.item()):.1f} "
                  f"| lr {lr:.2e} | gnorm {grad_norm:.3f} | {tok_per_sec:.0f} tok/s")
            t0 = time.time()

        # Validation
        if step % args.eval_interval == 0 and len(val_data) > block_size:
            val_loss = evaluate(model, val_data, block_size, args.batch_size, device)
            val_losses.append((step, val_loss))
            print(f"  val loss {val_loss:.4f} | val ppl {math.exp(val_loss):.1f}")

        # Checkpoint
        if step > 0 and step % args.save_interval == 0:
            save_checkpoint(model, optimizer, step, train_losses, val_losses,
                            CHECKPOINT_DIR / f"ckpt_{step:06d}.pt")

        # Sample
        if step % args.sample_interval == 0 and step > 0:
            model.eval()
            seed_text = "HAMLET:"
            ids = torch.tensor([tokenizer.encode(seed_text)], device=device)
            out = model.generate(ids, max_new_tokens=100, temperature=0.8, top_k=40, use_kv_cache=True)
            sample = tokenizer.decode(out[0].tolist())
            print(f"\n--- sample at step {step} ---\n{sample}\n{'---'*15}\n")
            model.train()

    # Final checkpoint
    save_checkpoint(model, optimizer, args.steps - 1, train_losses, val_losses,
                    CHECKPOINT_DIR / "ckpt_final.pt")

    if args.steps > 0:
        final_val = evaluate(model, val_data, block_size, args.batch_size, device)
        print(f"\nFinal val loss: {final_val:.4f} | ppl: {math.exp(final_val):.1f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Train the from-scratch transformer",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Data
    p.add_argument("--data", type=str, default=None, help="Path to training text file (downloads Shakespeare if omitted)")
    p.add_argument("--block_size", type=int, default=256, help="Context window / sequence length per batch")
    p.add_argument("--val_frac", type=float, default=0.1, help="Fraction of data for validation")

    # Model
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--d_ff", type=int, default=512)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--swiglu", action="store_true", default=True, help="Use SwiGLU FFN")
    p.add_argument("--dropout", type=float, default=0.1)

    # Training
    p.add_argument("--steps", type=int, default=3000, help="Total training steps")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    p.add_argument("--min_lr", type=float, default=3e-5, help="Minimum LR at end of cosine schedule")
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm")

    # Intervals
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--save_interval", type=int, default=1000)
    p.add_argument("--sample_interval", type=int, default=500)

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()

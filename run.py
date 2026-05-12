#!/usr/bin/env python3
"""
Generate text with the transformer.

Examples:
    python run.py
    python run.py --prompt "HAMLET:" --steps 200 --temperature 0.7 --top_k 40
    python run.py --ckpt checkpoints/ckpt_final.pt --prompt "To be"
    python run.py --d_model 256 --num_layers 6 --no-kv-cache --verbose
"""
import argparse
import time

import torch

from transformer import CharTokenizer, ModelConfig, Transformer


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model (ignored when --ckpt is given)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--d_ff", type=int, default=512)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--swiglu", action="store_true", default=True)

    # Generation
    p.add_argument("--prompt", type=str, default="Hello, world!")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--no-kv-cache", dest="use_kv_cache", action="store_false")

    # Misc
    p.add_argument("--ckpt", type=str, default=None, help="Load a trained checkpoint")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = CharTokenizer()

    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=device)
        cfg: ModelConfig = ckpt["cfg"]
        model = Transformer(cfg).to(device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded checkpoint: {args.ckpt}")
    else:
        cfg = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            max_seq_len=args.max_seq_len,
            use_swiglu=args.swiglu,
            dropout=0.0,
        )
        model = Transformer(cfg).to(device)

    print(f"Device: {device} | KV cache: {args.use_kv_cache} | temp: {args.temperature}")
    print(f'Prompt: "{args.prompt}"')
    print("=" * 70)

    ids = torch.tensor([tokenizer.encode(args.prompt)], device=device)

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=args.steps,
            temperature=args.temperature,
            top_k=args.top_k,
            use_kv_cache=args.use_kv_cache,
        )
    elapsed = time.time() - t0

    print(tokenizer.decode(out[0].tolist()))
    print("=" * 70)
    print(f"{args.steps} tokens in {elapsed:.2f}s  |  {args.steps / elapsed:.1f} tok/s  |  {elapsed * 1000 / args.steps:.1f} ms/tok")


if __name__ == "__main__":
    main()

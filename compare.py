#!/usr/bin/env python3
"""
KV cache vs no-cache speed comparison.

Demonstrates the O(T) vs O(T²) decode complexity difference:
  - Without cache: re-processes the entire growing context each step
  - With cache: processes only the new token, reuses stored K/V projections

Speedup scales with prompt length — longer prompts = bigger win.
"""
import argparse
import time

import torch

from transformer import CharTokenizer, ModelConfig, Transformer

PROMPT = (
    "what is going on in the barn? I hope that the cows and such are okay. "
    "What are you doing? Are you a pig or a cow? How are you doing? "
    "I am an animal! What kind of animal are you? I am a human! Yes I am!"
)


def run_comparison(prompt: str, steps: int, seed: int, cfg: ModelConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CharTokenizer()
    torch.manual_seed(seed)

    print(f"Device: {device} | seed: {seed}")
    print(f"Prompt length: {len(prompt)} chars | generating {steps} tokens")
    print("=" * 70)

    input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)

    # --- With KV cache ---
    model_kv = Transformer(cfg).to(device)
    t0 = time.time()
    with torch.no_grad():
        out_kv = model_kv.generate(input_ids, steps, temperature=0.8, use_kv_cache=True)
    kv_time = time.time() - t0
    kv_tps = steps / kv_time
    print(f"WITH cache:     {kv_time:.3f}s  |  {kv_tps:.1f} tok/s")
    print(f"  Output: {tokenizer.decode(out_kv[0].tolist())[:60]}...")

    # --- Without KV cache ---
    model_no = Transformer(cfg).to(device)
    t0 = time.time()
    with torch.no_grad():
        out_no = model_no.generate(input_ids, steps, temperature=0.8, use_kv_cache=False)
    no_time = time.time() - t0
    no_tps = steps / no_time
    print(f"\nWITHOUT cache:  {no_time:.3f}s  |  {no_tps:.1f} tok/s")
    print(f"  Output: {tokenizer.decode(out_no[0].tolist())[:60]}...")

    speedup = kv_tps / no_tps
    print(f"\nSpeedup: {speedup:.2f}x  (prompt_len={len(prompt)}, decode_steps={steps})")
    return speedup


def main():
    p = argparse.ArgumentParser(description="KV cache speed comparison",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--prompt", type=str, default=PROMPT)
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--d_ff", type=int, default=512)
    p.add_argument("--max_seq_len", type=int, default=2048)
    args = p.parse_args()

    cfg = ModelConfig(
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        use_swiglu=True,
        dropout=0.0,
    )
    run_comparison(args.prompt, args.steps, args.seed, cfg)


if __name__ == "__main__":
    main()

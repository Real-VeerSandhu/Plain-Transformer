#!/usr/bin/env python3
"""
Compare KV cache vs no-cache performance side by side.
"""
import time
import torch
from transformer.model import Transformer
from transformer.tokenizer import CharTokenizer

PROMPT = "what is going on in the barn? I hope that the cows and such are okay. What are you doing? Are you a pig or a cow? How are you doing? I am an animal! What kind of animal are you? I am a human! I am a human! Yes I am!"

def run_comparison(prompt=PROMPT * 1, steps=100, seed=42):
    """Run comparison between KV cache and no-cache."""
    
    # Configuration
    config = {
        'd_model': 128,
        'num_layers': 2,
        'num_heads': 4,
        'd_ff': 256,
        'max_seq_len': 2560,
        'use_swiglu': False,
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CharTokenizer()
    
    # Set random seed for reproducibility (or lack thereof)
    torch.manual_seed(seed)
    
    print(f"Device: {device}")
    print(f"Random seed: {seed}")
    print(f"Prompt: {prompt!r}")
    print(f"Generating {steps} tokens")
    print("=" * 80)
    
    # Test with KV cache
    print("WITH KV CACHE:")
    model_kv = Transformer(vocab_size=tokenizer.vocab_size, **config).to(device)
    input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    
    start_time = time.time()
    with torch.no_grad():
        output_kv = model_kv.generate(input_ids, steps, temperature=0.8, use_kv_cache=True)
    kv_time = time.time() - start_time
    
    kv_text = tokenizer.decode(output_kv[0].tolist())
    kv_tps = steps / kv_time
    
    print(f"  Time: {kv_time:.3f}s")
    print(f"  Tokens/sec: {kv_tps:.1f}")
    print(f"  Output: {kv_text[:50]}...")
    
    # Test without KV cache
    print("\nWITHOUT KV CACHE:")
    model_no_kv = Transformer(vocab_size=tokenizer.vocab_size, **config).to(device)
    
    start_time = time.time()
    with torch.no_grad():
        output_no_kv = model_no_kv.generate(input_ids, steps, temperature=0.8, use_kv_cache=False)
    no_kv_time = time.time() - start_time
    
    no_kv_text = tokenizer.decode(output_no_kv[0].tolist())
    no_kv_tps = steps / no_kv_time
    
    print(f"  Time: {no_kv_time:.3f}s")
    print(f"  Tokens/sec: {no_kv_tps:.1f}")
    print(f"  Output: {no_kv_text[:50]}...")
    
    # Show speedup
    speedup = kv_tps / no_kv_tps
    print("\n" + "=" * 80)
    print(f"SPEEDUP: {speedup:.2f}x faster with KV cache")
    print(f"KV cache is {speedup:.1f}x faster")
    
    return speedup

if __name__ == "__main__":
    import random
    
    # Test with different seeds to show different random outputs
    print("Testing different random seeds:")
    print("-" * 40)
    
    # # Test with default seed
    # run_comparison(seed=42)
    
    # print("\n" + "="*80 + "\n")
    
    # # Test with different seed
    # run_comparison(seed=123)
    
    print("\n" + "="*80 + "\n")
    
    # Test with random seed
    random_seed = random.randint(1, 10000)
    run_comparison(seed=3)

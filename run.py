#!/usr/bin/env python3
"""
Run a transformer model with various configuration options.
"""
import argparse
import time
import torch

from transformer.model import Transformer
from transformer.tokenizer import CharTokenizer

def main():
    parser = argparse.ArgumentParser(
        description="Run a transformer model with configurable options",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument("--d_model", type=int, default=128, help="Model dimension")
    model_group.add_argument("--num_layers", type=int, default=2, help="Number of transformer layers")
    model_group.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    model_group.add_argument("--d_ff", type=int, default=256, help="Feed-forward dimension")
    model_group.add_argument("--max_seq_len", type=int, default=256, help="Maximum sequence length")
    model_group.add_argument("--swiglu", action="store_true", help="Use SwiGLU activation")
    
    # Generation parameters
    gen_group = parser.add_argument_group('Generation Parameters')
    gen_group.add_argument("--prompt", type=str, default="Hello, world!", help="Input prompt")
    gen_group.add_argument("--steps", type=int, default=50, help="Number of tokens to generate")
    gen_group.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (lower=more deterministic)")
    gen_group.add_argument("--top_k", type=int, default=None, help="Top-k sampling (restrict vocabulary)")
    
    # Performance options
    perf_group = parser.add_argument_group('Performance Options')
    perf_group.add_argument("--no-kv-cache", dest="use_kv_cache", action="store_false", 
                           help="Disable KV caching (slower but uses less memory)")
    perf_group.add_argument("--verbose", action="store_true", help="Show timing information")
    
    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize tokenizer and model
    tokenizer = CharTokenizer()
    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        use_swiglu=args.swiglu,
    ).to(device)
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    print(f"Model config: L={args.num_layers}, d_model={args.d_model}, H={args.num_heads}")
    
    # Encode input prompt
    input_ids = torch.tensor([tokenizer.encode(args.prompt)], device=device)
    print(f"Input: {args.prompt!r}")
    print("=" * 80)
    
    # Generate tokens
    print(f"Generating {args.steps} tokens with{'out' if not args.use_kv_cache else ''} KV cache...")
    
    if args.use_kv_cache:
        # Prefill timing
        prefill_start = time.time()
        with torch.no_grad():
            _ = model(input_ids, use_kv_cache=True, start_pos=0)
        prefill_time = time.time() - prefill_start
        
        # Decode timing
        decode_start = time.time()
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_new_tokens=args.steps,
                temperature=args.temperature,
                top_k=args.top_k,
                use_kv_cache=True,
            )
        decode_time = time.time() - decode_start
        
        total_time = prefill_time + decode_time
        
        if args.verbose:
            print(f"  Prefill time: {prefill_time*1000:.2f}ms")
            print(f"  Decode time: {decode_time*1000:.2f}ms")
            print(f"  Total time: {total_time*1000:.2f}ms")
    else:
        # No cache timing
        start_time = time.time()
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_new_tokens=args.steps,
                temperature=args.temperature,
                top_k=args.top_k,
                use_kv_cache=False,
            )
        total_time = time.time() - start_time
    
    # Decode and print the generated text
    generated_text = tokenizer.decode(generated[0].tolist())
    print("\nGenerated text:")
    print("-" * 80)
    print(generated_text)
    print("-" * 80)
    
    # Print performance metrics
    tokens_per_sec = args.steps / total_time
    ms_per_token = total_time * 1000 / args.steps
    
    print(f"\nGeneration stats:")
    print(f"Time: {total_time:.2f}s")
    print(f"Tokens per second: {tokens_per_sec:.1f}")
    print(f"ms/token: {ms_per_token:.1f}")
    print(f"KV cache: {'enabled' if args.use_kv_cache else 'disabled'}")
    print(f"Temperature: {args.temperature}")
    print(f"Seed: {args.seed}")

if __name__ == "__main__":
    main()

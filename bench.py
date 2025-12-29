#!/usr/bin/env python3
"""
Benchmark script to compare KV cache vs no-cache performance across different model sizes and sequence lengths.
"""
import time
import torch
from typing import List, Dict, Tuple

from transformer.model import Transformer
from transformer.tokenizer import CharTokenizer

def benchmark_model(
    model_config: Dict,
    prompt_lengths: List[int],
    num_decode_tokens: int = 50,
    use_kv_cache: bool = True,
    num_runs: int = 3,
) -> Dict:
    """
    Benchmark a model configuration.
    
    Args:
        model_config: Dictionary with model configuration
        prompt_lengths: List of prompt lengths to test
        num_decode_tokens: Number of tokens to decode
        use_kv_cache: Whether to use KV cache
        num_runs: Number of runs to average over
        
    Returns:
        Dictionary with benchmark results
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CharTokenizer()
    
    # Create model
    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        **model_config
    ).to(device)
    
    results = {
        'model_config': model_config,
        'use_kv_cache': use_kv_cache,
        'results': {}
    }
    
    for prompt_len in prompt_lengths:
        print(f"  Testing prompt length {prompt_len}...")
        
        # Skip if prompt + decode would exceed max_seq_len
        if prompt_len + num_decode_tokens > model_config['max_seq_len']:
            print(f"    Skipping: prompt_len ({prompt_len}) + decode_tokens ({num_decode_tokens}) > max_seq_len ({model_config['max_seq_len']})")
            continue
        
        # Create prompt
        prompt = "x" * prompt_len  # Use simple repeated character
        input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
        
        prefill_times = []
        decode_times = []
        
        for run in range(num_runs):
            # Clear cache before each run
            model.clear_cache()
            
            # Measure prefill time
            if use_kv_cache:
                start_time = time.time()
                with torch.no_grad():
                    _ = model(input_ids, use_kv_cache=True, start_pos=0)
                prefill_time = time.time() - start_time
                prefill_times.append(prefill_time)
                
                # Measure decode time
                start_time = time.time()
                with torch.no_grad():
                    _ = model.generate(
                        input_ids,
                        max_new_tokens=num_decode_tokens,
                        temperature=1.0,
                        use_kv_cache=True
                    )
                decode_time = time.time() - start_time
                decode_times.append(decode_time)
            else:
                # No cache: measure total time
                start_time = time.time()
                with torch.no_grad():
                    _ = model.generate(
                        input_ids,
                        max_new_tokens=num_decode_tokens,
                        temperature=1.0,
                        use_kv_cache=False
                    )
                total_time = time.time() - start_time
                
                # For no-cache, we'll estimate prefill as part of first token
                prefill_times.append(total_time / num_decode_tokens)  # Rough estimate
                decode_times.append(total_time / num_decode_tokens)  # Average per token
        
        # Store results
        results['results'][prompt_len] = {
            'prefill_time_ms': sum(prefill_times) / len(prefill_times) * 1000,
            'decode_time_ms_per_token': sum(decode_times) / len(decode_times) * 1000 / num_decode_tokens,
            'tokens_per_second': num_decode_tokens / (sum(decode_times) / len(decode_times))
        }
    
    return results

def print_benchmark_results(all_results: List[Dict]):
    """Print benchmark results in a formatted table."""
    print("\n" + "="*100)
    print("BENCHMARK RESULTS")
    print("="*100)
    
    for result in all_results:
        config = result['model_config']
        kv_cache = "KV Cache" if result['use_kv_cache'] else "No Cache"
        
        print(f"\nModel: L={config['num_layers']}, d_model={config['d_model']}, H={config['num_heads']} | {kv_cache}")
        print("-" * 80)
        print(f"{'Prompt Len':<12} {'Prefill (ms)':<12} {'Decode (ms/tok)':<15} {'Tokens/sec':<12}")
        print("-" * 80)
        
        for prompt_len, metrics in result['results'].items():
            print(f"{prompt_len:<12} {metrics['prefill_time_ms']:<12.2f} "
                  f"{metrics['decode_time_ms_per_token']:<15.2f} {metrics['tokens_per_second']:<12.1f}")

def main():
    # Model configurations
    configs = [
        {'num_layers': 2, 'd_model': 128, 'num_heads': 4, 'd_ff': 256, 'max_seq_len': 1024, 'use_swiglu': False},  # Small
        {'num_layers': 4, 'd_model': 256, 'num_heads': 8, 'd_ff': 512, 'max_seq_len': 1024, 'use_swiglu': False},  # Medium
    ]
    
    # Prompt lengths to test (ensure they don't exceed max_seq_len when combined with decode tokens)
    prompt_lengths = [16, 32, 64, 128, 256, 512]  # Added back 512 since we increased max_seq_len
    
    # Run benchmarks
    all_results = []
    
    for config in configs:
        print(f"\nBenchmarking model: L={config['num_layers']}, d_model={config['d_model']}, H={config['num_heads']}")
        
        # Test with KV cache
        print("  With KV cache:")
        kv_results = benchmark_model(config, prompt_lengths, use_kv_cache=True)
        all_results.append(kv_results)
        
        # Test without KV cache
        print("  Without KV cache:")
        no_kv_results = benchmark_model(config, prompt_lengths, use_kv_cache=False)
        all_results.append(no_kv_results)
    
    # Print results
    print_benchmark_results(all_results)
    
    print("\n" + "="*100)

if __name__ == "__main__":
    main()

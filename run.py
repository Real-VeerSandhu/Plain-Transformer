import time
import torch

from transformer.model import Transformer
from transformer.tokenizer import CharTokenizer

def main():
    # Configuration
    config = {
        'd_model': 128,
        'num_layers': 2,
        'num_heads': 4,
        'd_ff': 256,
        'max_seq_len': 256,
        'use_swiglu': False,
        'prompt': "Hello, world!",
        'steps': 50,
        'temperature': 0.8,
        'top_k': None,
        'use_kv_cache': True,
        'seed': 42
    }
    
    # Set random seed for reproducibility
    torch.manual_seed(config['seed'])
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize tokenizer and model
    tokenizer = CharTokenizer()
    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        max_seq_len=config['max_seq_len'],
        use_swiglu=config['use_swiglu'],
    ).to(device)
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Encode input prompt
    input_ids = torch.tensor([tokenizer.encode(config['prompt'])], device=device)
    print(f"Input: {config['prompt']!r}")
    print("=" * 80)
    
    # Generate tokens
    print(f"Generating {config['steps']} tokens with{'out' if not config['use_kv_cache'] else ''} KV cache...")
    start_time = time.time()
    
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_new_tokens=config['steps'],
            temperature=config['temperature'],
            top_k=config['top_k'],
            use_kv_cache=config['use_kv_cache'],
        )
    
    generation_time = time.time() - start_time
    
    # Decode and print the generated text
    generated_text = tokenizer.decode(generated[0].tolist())
    print("\nGenerated text:")
    print("-" * 80)
    print(generated_text)
    print("-" * 80)
    
    # Print performance metrics
    tokens_per_sec = config['steps'] / generation_time
    ms_per_token = generation_time * 1000 / config['steps']
    
    print(f"\nGeneration stats:")
    print(f"Time: {generation_time:.2f}s")
    print(f"Tokens per second: {tokens_per_sec:.1f}")
    print(f"ms/token: {ms_per_token:.1f}")
    print(f"KV cache: {'enabled' if config['use_kv_cache'] else 'disabled'}")

if __name__ == "__main__":
    main()

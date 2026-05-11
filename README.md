# Plain-Transformer

A Transformer in PyTorch with causal attention, RoPE, and RMSNorm over a configurable architecture. Designed as a hands-on testbed for model internals, inference, and training dynamics.

## Features

- **Decoder-only transformer** with causal attention
- **KV caching** for efficient generation (12x+ speedup on long sequences)
- **Rotary Position Embeddings (RoPE)** for better positional encoding
- **Configurable architecture** - change model size, layers, heads
- **Character-level tokenizer** (256 vocab)
- **Performance benchmarking** - compare cache vs no-cache


## Project Structure

```
transformer/
├── __init__.py
├── attention.py    # Multi-head attention with RoPE
├── block.py        # Transformer blocks
├── ffn.py          # Feed-forward networks
├── model.py        # Main transformer model
├── rope.py         # Rotary position embeddings
└── tokenizer.py    # Character-level tokenizer
run.py              # Main execution script
compare.py          # KV cache comparison
bench.py            # Performance benchmarking
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run with default settings
python run.py

# Test KV cache speedup
python compare.py

# Benchmark across model sizes
python bench.py
```

## Command Line Options

### Model Configuration
- `--d_model`: Model dimension (default: 128)
- `--num_layers`: Number of layers (default: 2)  
- `--num_heads`: Number of attention heads (default: 4)
- `--d_ff`: Feed-forward dimension (default: 256)
- `--max_seq_len`: Maximum sequence length (default: 256)
- `--swiglu`: Use SwiGLU activation

### Generation Parameters
- `--prompt`: Input prompt (default: "Hello, world!")
- `--steps`: Tokens to generate (default: 50)
- `--temperature`: Sampling temperature (default: 0.8)
- `--top_k`: Top-k sampling

### Performance Options
- `--no-kv-cache`: Disable KV caching (slower but uses less memory)
- `--verbose`: Show detailed timing information
- `--seed`: Random seed (default: 42)

## Examples

```bash
# Small model, no cache
python run.py --d_model 64 --num_layers 2 --no-kv-cache

# Larger model, high temperature
python run.py --d_model 256 --num_layers 4 --temperature 1.2

# Custom prompt with detailed timing
python run.py --prompt "Hello world. Can you predict the next word by looking at the previous words?" --verbose

# Benchmark comparison
python compare.py --seed 123
```

## Architecture

The model uses standard transformer components:
- Multi-head self-attention with RoPE
- RMSNorm layer normalization  
- Feed-forward network (GELU/SwiGLU)
- Token embeddings + positional embeddings

## Notes

- Weights are randomly initialized (not trained)
- Output is random but deterministic for same seed
- KV cache benefits scale with sequence length
- CPU-optimized for small models

import math
import torch
import torch.nn as nn

def apply_rope(x: torch.Tensor, head_dim: int, start_pos: int = 0) -> torch.Tensor:
    """
    Apply Rotary Position Embedding (RoPE) to query and key tensors.
    
    Args:
        x: Input tensor of shape (batch_size, num_heads, seq_len, head_dim)
        head_dim: Dimension of each attention head
        start_pos: Starting position for the sequence
        
    Returns:
        RoPE-applied tensor of the same shape
    """
    batch_size, num_heads, seq_len, _ = x.shape
    device = x.device
    
    # Create position indices
    pos = torch.arange(start_pos, start_pos + seq_len, device=device, dtype=torch.float32)
    
    # Compute the frequencies
    dim = head_dim
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
    
    # Compute frequency matrix
    freqs = torch.outer(pos, inv_freq)  # (seq_len, dim/2)
    
    # Compute sin and cos
    sin = torch.sin(freqs)  # (seq_len, dim/2)
    cos = torch.cos(freqs)  # (seq_len, dim/2)
    
    # Expand to match tensor dimensions
    sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim/2)
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim/2)
    
    # Apply RoPE
    x_rotated = torch.zeros_like(x)
    
    # Split into real and imaginary parts
    x_real = x[..., :dim//2]
    x_imag = x[..., dim//2:]
    
    # Apply rotation
    x_rotated[..., :dim//2] = x_real * cos - x_imag * sin
    x_rotated[..., dim//2:] = x_real * sin + x_imag * cos
    
    return x_rotated

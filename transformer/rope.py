"""Rotary Position Embeddings (RoPE).

Key idea: instead of adding a position vector, rotate Q and K in 2D subspaces
by an angle proportional to position. This makes attention scores sensitive to
*relative* distance rather than absolute position.
"""
import torch


def build_rope_cache(seq_len: int, head_dim: int, device: torch.device, base: float = 10000.0):
    """
    Pre-compute sin/cos tables for RoPE up to seq_len positions.
    Returns (cos, sin) each of shape (seq_len, head_dim/2).
    """
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, inv_freq)  # (seq_len, half)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, head_dim: int, start_pos: int = 0) -> torch.Tensor:
    """
    Apply RoPE to x of shape (B, num_heads, T, head_dim).

    Splits head_dim into two halves and applies rotation:
        [x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]

    This is equivalent to multiplying complex numbers e^{i*theta} * (x0 + i*x1).
    """
    B, H, T, D = x.shape
    cos, sin = build_rope_cache(start_pos + T, head_dim, x.device)
    cos = cos[start_pos:start_pos + T]  # (T, D/2)
    sin = sin[start_pos:start_pos + T]

    # Broadcast to (1, 1, T, D/2)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    x0 = x[..., : D // 2]
    x1 = x[..., D // 2 :]

    x_rot = torch.cat([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)
    return x_rot

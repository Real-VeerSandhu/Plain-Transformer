"""Transformer block with pre-norm residual connections."""
import torch
import torch.nn as nn
from typing import Optional, Tuple

from .attention import MultiHeadAttention
from .ffn import FeedForward, SwiGLU


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no mean-centering, just scale)."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * self._norm(x.float()).type_as(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LayerNorm before each sub-layer."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, use_swiglu: bool = False, dropout: float = 0.0):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff) if use_swiglu else FeedForward(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_kv_cache: bool = False,
        start_pos: int = 0,
        return_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_out, attn_weights = self.attn(
            self.ln1(x), mask=mask, use_kv_cache=use_kv_cache,
            start_pos=start_pos, return_weights=return_attn_weights,
        )
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x, attn_weights

    def clear_cache(self):
        self.attn.clear_cache()

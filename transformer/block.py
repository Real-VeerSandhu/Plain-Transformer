import torch
import torch.nn as nn
from typing import Optional

from .attention import MultiHeadAttention, create_causal_mask
from .ffn import FeedForward, SwiGLU

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * self._norm(x.float()).type_as(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, use_swiglu: bool = False):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ln2 = RMSNorm(d_model)
        
        # Choose between standard FFN and SwiGLU
        if use_swiglu:
            self.ffn = SwiGLU(d_model, d_ff)
        else:
            self.ffn = FeedForward(d_model, d_ff)
            
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_kv_cache: bool = False,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            use_kv_cache: Whether to use KV caching
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Self-attention with residual connection
        attn_out = self.attn(self.ln1(x), mask, use_kv_cache, start_pos)
        x = x + attn_out
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(self.ln2(x))
        x = x + ffn_out
        
        return x
    
    def clear_cache(self):
        """Clear the KV cache in the attention layer"""
        self.attn.clear_cache()

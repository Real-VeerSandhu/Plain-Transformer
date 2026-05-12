"""Multi-head causal self-attention with RoPE and KV caching."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .rope import apply_rope


class MultiHeadAttention(nn.Module):
    """
    Causal multi-head self-attention.

    Supports:
    - Rotary Position Embeddings (RoPE) on Q and K
    - KV caching for autoregressive decoding
    - Attention weight capture for inspection
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)

        self.k_cache: Optional[torch.Tensor] = None
        self.v_cache: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_kv_cache: bool = False,
        start_pos: int = 0,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, _ = x.shape

        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        # Reshape to (B, num_heads, T, head_dim)
        def split_heads(t):
            return t.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        Q = split_heads(Q)
        K = split_heads(K)
        V = split_heads(V)

        # Apply RoPE positional encoding to Q and K
        Q = apply_rope(Q, self.head_dim, start_pos)
        K = apply_rope(K, self.head_dim, start_pos)

        # KV cache: append new K/V to cached history then store
        if use_kv_cache:
            if self.k_cache is not None:
                K = torch.cat([self.k_cache, K], dim=2)
                V = torch.cat([self.v_cache, V], dim=2)
            self.k_cache = K
            self.v_cache = V

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, H, T_q, T_k)

        if mask is not None:
            T_k = scores.size(-1)
            T_q = scores.size(-2)
            # mask is (T_q, T_k) or broadcastable — True means keep
            m = mask[:T_q, :T_k] if mask.dim() == 2 else mask
            scores = scores.masked_fill(~m.bool(), float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        out = torch.matmul(attn_weights, V)  # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        out = self.Wo(out)

        weights = attn_weights.detach() if return_weights else None
        return out, weights

    def clear_cache(self):
        self.k_cache = None
        self.v_cache = None

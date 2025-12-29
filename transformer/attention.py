import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        
        # For caching during generation
        self.k_cache = None
        self.v_cache = None
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_kv_cache: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor of shape (batch_size, seq_len, seq_len)
            use_kv_cache: Whether to use KV caching
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project queries, keys, and values
        Q = self.Wq(x)  # (batch_size, seq_len, d_model)
        K = self.Wk(x)  # (batch_size, seq_len, d_model)
        V = self.Wv(x)  # (batch_size, seq_len, d_model)
        
        if use_kv_cache and self.k_cache is not None and self.v_cache is not None:
            # Append new K, V to cache
            K = torch.cat([self.k_cache, K], dim=1)
            V = torch.cat([self.v_cache, V], dim=1)
            
        # Store K, V in cache for next step
        if use_kv_cache:
            self.k_cache = K
            self.v_cache = V
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        # Q, K, V shapes: (batch_size, num_heads, seq_len, head_dim)
        # scores shape: (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            # Get the current sequence length from the scores tensor
            seq_len = scores.size(-1)
            
            # Create a causal mask if needed
            if mask.dim() == 4:  # (batch_size, 1, seq_len, seq_len)
                mask = mask[:, :, :seq_len, :seq_len]
            elif mask.dim() == 3:  # (1, seq_len, seq_len)
                mask = mask[:, :seq_len, :seq_len]
                mask = mask.unsqueeze(1)  # Add head dimension
            elif mask.dim() == 2:  # (seq_len, seq_len)
                mask = mask[:seq_len, :seq_len]
                mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
            
            # Ensure the mask is on the same device as scores
            mask = mask.to(scores.device)
            
            # Apply the mask
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax to get attention weights
        attention = F.softmax(scores, dim=-1)
        output = torch.matmul(attention, V)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Concatenate heads and project back to d_model
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.Wo(output)  # (batch_size, seq_len, d_model)
    
    def clear_cache(self):
        """Clear the KV cache"""
        self.k_cache = None
        self.v_cache = None


def create_causal_mask(seq_len: int, device: str = 'cpu') -> torch.Tensor:
    """Create a causal mask for self-attention"""
    return torch.tril(torch.ones(seq_len, seq_len, device=device)).view(1, 1, seq_len, seq_len)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from .block import TransformerBlock, RMSNorm

class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        d_ff: int = 256,
        max_seq_len: int = 256,
        use_swiglu: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, use_swiglu)
            for _ in range(num_layers)
        ])
        
        # Final layer norm and output projection
        self.ln_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights (optional but common in many models)
        self.token_embedding.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        use_kv_cache: bool = False,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            targets: Optional target token IDs of shape (batch_size, seq_len)
            use_kv_cache: Whether to use KV caching
            
        Returns:
            logits: Output logits of shape (batch_size, seq_len, vocab_size)
            loss: Optional loss value if targets are provided
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create position ids and embeddings
        if start_pos < 0:
            raise ValueError("start_pos must be >= 0")
        if start_pos + seq_len > self.max_seq_len:
            raise ValueError("start_pos + seq_len exceeds max_seq_len")

        pos = torch.arange(start_pos, start_pos + seq_len, dtype=torch.long, device=device).unsqueeze(0)  # (1, seq_len)
        
        # Get token and position embeddings
        tok_emb = self.token_embedding(input_ids)  # (batch_size, seq_len, d_model)
        pos_emb = self.pos_embedding[:, start_pos : start_pos + seq_len, :]  # (1, seq_len, d_model)
        
        # Combine embeddings
        x = tok_emb + pos_emb
        
        # Create causal mask for self-attention
        # We need to create a mask where each position (i, j) is True if j <= i (causal)
        if seq_len > 1:
            # Create a lower triangular matrix of ones
            mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
            mask = torch.tril(mask)  # Lower triangular mask
            
            # Add batch dimension (but not head dimension yet, as it will be added in attention)
            # Shape: (1, seq_len, seq_len)
            mask = mask.unsqueeze(0)
        else:
            mask = None
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask, use_kv_cache, start_pos)
        
        # Final layer norm and projection to vocab size
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)
        
        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1  # ignore padding tokens if any
            )
            
        return logits, loss
    
    def clear_cache(self):
        """Clear the KV cache in all transformer blocks"""
        for block in self.blocks:
            block.clear_cache()
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        use_kv_cache: bool = True,
    ) -> torch.Tensor:
        """
        Generate new tokens using the model.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from the top k most likely tokens
            use_kv_cache: Whether to use KV caching for faster generation
            
        Returns:
            Generated token IDs of shape (batch_size, seq_len + max_new_tokens)
        """
        self.eval()
        self.clear_cache()

        # Keep track of all generated tokens
        generated = input_ids

        if use_kv_cache:
            prompt_len = input_ids.size(1)
            if prompt_len > self.max_seq_len:
                raise ValueError("prompt length exceeds max_seq_len")
            if prompt_len + max_new_tokens > self.max_seq_len:
                raise ValueError("prompt_len + max_new_tokens exceeds max_seq_len")

            # For very short sequences, KV cache might not be beneficial
            # But we'll still use it for demonstration
            if prompt_len > 1:
                # Prefill: run the full prompt once to populate KV caches.
                _logits, _loss = self(input_ids, use_kv_cache=True, start_pos=0)
            else:
                # For single token, no need to prefill
                pass

            # Decode: feed exactly 1 new token per step so cache growth is correct.
            for step in range(max_new_tokens):
                last_token = generated[:, -1:]  # (B, 1)
                logits, _ = self(last_token, use_kv_cache=True, start_pos=prompt_len + step)

                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float('Inf')

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)

            self.clear_cache()
            return generated

        for _ in range(max_new_tokens):
            # Get the relevant portion of the sequence
            seq_len = generated.size(1)
            if seq_len > self.max_seq_len:
                # If sequence is too long, use the most recent tokens
                context = generated[:, -self.max_seq_len:]
            else:
                context = generated

            # Get model predictions
            logits, _ = self(context, use_kv_cache=False, start_pos=0)
            
            # Get the logits for the last token
            logits = logits[:, -1, :] / temperature
            
            # Optionally restrict to top-k tokens
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append the sampled token to the sequence
            generated = torch.cat([generated, next_token], dim=1)
        
        self.clear_cache()
        return generated

"""Decoder-only transformer model."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .block import TransformerBlock, RMSNorm
from .config import ModelConfig


class Transformer(nn.Module):
    """
    Decoder-only transformer.

    Architecture (per block):
        x -> RMSNorm -> CausalMHA (RoPE) -> residual
          -> RMSNorm -> FFN (SwiGLU or GELU) -> residual
    Final: RMSNorm -> lm_head

    No learned positional embeddings — position information comes solely from RoPE
    applied inside each attention layer.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.num_heads, cfg.d_ff, cfg.use_swiglu, cfg.dropout)
            for _ in range(cfg.num_layers)
        ])
        self.ln_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying: token embedding and lm_head share parameters
        self.token_embedding.weight = self.lm_head.weight

        self._init_weights()

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Transformer: {n_params:,} params | L={cfg.num_layers} d={cfg.d_model} H={cfg.num_heads} FFN={cfg.d_ff}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Lower-triangular boolean mask: True = attend, False = mask out."""
        return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        use_kv_cache: bool = False,
        start_pos: int = 0,
        return_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[Optional[torch.Tensor]]]:
        """
        Args:
            input_ids:  (B, T) token ids
            targets:    (B, T) for computing cross-entropy loss
            use_kv_cache: enable autoregressive KV cache
            start_pos:  position offset when decoding with cache
            return_attn_weights: collect attention maps from all layers

        Returns:
            logits:       (B, T, vocab_size)
            loss:         scalar or None
            attn_weights: list of (B, H, T, T) tensors per layer, or empty list
        """
        B, T = input_ids.shape

        if start_pos + T > self.cfg.max_seq_len:
            raise ValueError(f"start_pos ({start_pos}) + seq_len ({T}) > max_seq_len ({self.cfg.max_seq_len})")

        x = self.token_embedding(input_ids)  # (B, T, d_model)

        mask = self._causal_mask(T, input_ids.device) if T > 1 else None

        all_attn_weights = []
        for block in self.blocks:
            x, w = block(x, mask=mask, use_kv_cache=use_kv_cache,
                         start_pos=start_pos, return_attn_weights=return_attn_weights)
            all_attn_weights.append(w)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss, all_attn_weights

    def clear_cache(self):
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
        """Autoregressive token generation."""
        self.eval()
        self.clear_cache()
        generated = input_ids

        if use_kv_cache:
            prompt_len = input_ids.size(1)
            if prompt_len + max_new_tokens > self.cfg.max_seq_len:
                raise ValueError("prompt_len + max_new_tokens exceeds max_seq_len")

            # Prefill: process the entire prompt to populate KV caches
            if prompt_len > 1:
                self(input_ids, use_kv_cache=True, start_pos=0)

            # Decode: one token at a time
            for step in range(max_new_tokens):
                last = generated[:, -1:]
                logits, _, _ = self(last, use_kv_cache=True, start_pos=prompt_len + step)
                next_token = self._sample(logits[:, -1, :], temperature, top_k)
                generated = torch.cat([generated, next_token], dim=1)
        else:
            for _ in range(max_new_tokens):
                ctx = generated[:, -self.cfg.max_seq_len:]
                logits, _, _ = self(ctx, use_kv_cache=False, start_pos=0)
                next_token = self._sample(logits[:, -1, :], temperature, top_k)
                generated = torch.cat([generated, next_token], dim=1)

        self.clear_cache()
        return generated

    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float, top_k: Optional[int]) -> torch.Tensor:
        logits = logits / max(temperature, 1e-8)
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @classmethod
    def from_config(cls, **kwargs) -> "Transformer":
        """Convenience constructor: Transformer.from_config(d_model=256, ...)."""
        return cls(ModelConfig(**kwargs))

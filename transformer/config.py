from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 256
    d_model: int = 128
    num_layers: int = 4
    num_heads: int = 4
    d_ff: int = 512
    max_seq_len: int = 512
    use_swiglu: bool = True
    dropout: float = 0.1

    def __post_init__(self):
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        assert self.d_ff > self.d_model, "d_ff should be larger than d_model"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

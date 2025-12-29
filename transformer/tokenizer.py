import numpy as np
from typing import List, Dict, Tuple

class CharTokenizer:
    def __init__(self):
        self.vocab_size = 256  # ASCII
        self.stoi = {chr(i): i for i in range(self.vocab_size)}
        self.itos = {i: chr(i) for i in range(self.vocab_size)}
        
    def encode(self, text: str) -> List[int]:
        """Convert string to list of token IDs"""
        return [self.stoi[char] for char in text]
    
    def decode(self, tokens: List[int]) -> str:
        """Convert list of token IDs back to string"""
        return ''.join([self.itos[token] for token in tokens])
    
    def __call__(self, text: str) -> List[int]:
        return self.encode(text)

"""
GPT-2 Model Configuration
"""
from dataclasses import dataclass

@dataclass
class GPT2Config:
    block_size: int = 1024 #context window
    vocab_size: int = 50257 #GPT2 vocabulary size
    n_layer : int = 12 #number of layers
    n_head : int = 12 #number of heads
    n_embd : int = 768 #embedding dimension


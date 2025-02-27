"""
GPT-2 Model Configuration
"""
from dataclasses import dataclass

@dataclass
class GPT2Config:
    block_size: int = 1024 #maximum context length
    vocab_size: int = 50257 #number of unique tokens in the GPT2 vocabulary
    n_layer : int = 12 #number of transformer layers
    n_head : int = 12 #number of attention heads within an attention layer
    n_embd : int = 768 #dimensionality of token embeddings


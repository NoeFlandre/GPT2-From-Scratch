"""
Transformer Block
"""
import torch.nn as nn
from model.attention import CausalSelfAttention
from model.mlp import MLP

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x)) 
        x = x + self.mlp(self.ln2(x))
        return x
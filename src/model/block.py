"""
Transformer Block
"""
import torch.nn as nn
from src.model.attention import CausalSelfAttention
from src.model.mlp import MLP

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd) # layer norm before the attention mechanism
        self.attn = CausalSelfAttention(config) # masked multihead self attention
        self.ln2 = nn.LayerNorm(config.n_embd) # layer norm before MLP
        self.mlp = MLP(config) #feedforward neural network

    # defines how is the input processed throughout the transformer
    def forward(self, x):
        x = x + self.attn(self.ln1(x)) # residual connection + attention(layer norm)
        x = x + self.mlp(self.ln2(x)) # residual connection + MLP(layer norm)
        return x
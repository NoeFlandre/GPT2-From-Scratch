"""
Masked MultiHead Self Attention
"""

import torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0 # checking if the embedding dimension is divisible by the number of attention heads
        self.attn = nn.Linear(config.n_embd, 3 * config.n_embd) # Linear layer to project the input tensor to a 3D tensor which will later define the query, key and value
        self.proj = nn.Linear(config.n_embd, config.n_embd) # Linear layer to project the output of the attention layer back to the embedding dimension
        self.n_head = config.n_head # number of attention heads
        self.n_embd = config.n_embd # embedding dimension
    
    def forward(self, x):
        B, T, C = x.size() # B = batch size, T = sequence length, C = embedding dimension
        qkv = self.attn(x) # 
        q, k, v = qkv.split(self.embd, dim=2) # split the 3D tensor into 3 2D tensors (Queries, Keys, Values)

        #allows parallel computation of the attention mechanism across multiple heads
        k = k.view(B, T, self.n_head, C // self.n_embd).transpose(1, 2) # reshape the keys tensor to (B, T, n_head, head_dim) and transpose it to (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, C // self.n_embd).transpose(1, 2) # reshape the queries tensor to (B, T, n_head, head_dim) and transpose it to (B, n_head, T, head_dim)
        v = v.view(B, T, self.n_head, C // self.n_embd).transpose(1, 2) # reshape the values tensor to (B, T, n_head, head_dim) and transpose it to (B, n_head, T, head_dim)

        # using flash attention for efficiency (kernel fusion)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # apply scaled dot product attention and applies masking to avoid processing future tokens

        y = y.transpose(1, 2).contiguous().view(B, T, C) # transpose the output tensor to (B, n_head, T, head_dim) and reshape it to (B, T, C)
        y = self.proj(y) # project the output tensor to the embedding dimension

        return y
"""
GPT2 Model Implementation
"""

import torch.nn as nn
from model.block import Block
from model.config import GPT2Config

class GPT2(nn.Module):
    def __init__(self, config): # call the constructor of nn.Module
        self.config = config

        self.transformer = nn.ModuleDict(dict( # we organize our submodules using a dictionary
            wte = nn.Embedding(config.vocab_size, config.n_embd), # embedding converting each word into a dense vector
            wpe = nn.Embedding(config.block_size, config.n_embd), # positional embedding to keep track of the sequential information
            h = nn.ModuleList([Block(config) for i in range(config.n_layer)]), # number of transformer blocks 
            ln_f = nn.LayerNorm(config.n_embd), # final layer norm
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False) # fully connected layer mapping the embedding dimension back to the vocabulary size

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight
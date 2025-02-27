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

        self.apply(self.init_weights) #initialize the weights

    # Function to initialize the weights according to the GPT paper. We don't change the Layer Norm initialization as Pytorch is already handling it right. 
    # One common indicator for the initial standard deviation is 1/sqrt(number of features in the incoming layer)
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02

            if hasattr(module, 'WEIGHT_SCALE_INIT'):
                std *= (2*self.config.n_layer)**-0.5 # two blocks are adding to the residual pathway (the attention and the MLP, hence the factor 2). To avoid the variance growing too much, we are initializing with 1/sqrt(N)
            nn.init.normal_(module.weight, mean=0.0, std = std) #initialize the weights using a normal distribution

            if module.bias is not None:
                nn.init.zeros_(module.bias) # initialize the biases with zeros

        elif isinstance(module, nn.Embedding): 
            nn.init.normal_(module.weight, mean = 0.0, std = 0.02) # initialize the embeddings with a normal distribution
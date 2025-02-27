"""
GPT2 Model Implementation
"""
import torch
import torch.nn as nn
from model.block import Block
from model.config import GPT2Config
from torch.nn import functional as F
import inspect

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
    
    def forward(self, idx, targets = None):
        #idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is {self.config.block_size}"

        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device = idx.device) 
        pos_emb = self.transformer.wpe(pos) # position embeddings 
        tok_emb = self.tranformer.wte(idx) # token embeddings 
        x = pos_emb + tok_emb # summing the position and token embeddings

        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


    # weight decay is used to force the model to distribute the work to multiple parameters and not allow certain parameters to get extensively too important, it's a form of regularization.
    # we only decay the embeddings and the matmul parameters, not the LayerNorms and Biases

    def configure_optimizers(self, weight_decay, learning_rate, device):
          # we are starting with all the parameters requiring gradient
          para_dict = {pn: p for pn, p in self.named_parameters()}
          para_dict = {pn: p for pn, p in para_dict.items() if p.requires_grad}

          # we are creating optim groups. Only 2D parameters will be weight decayed
          decay_params = [p for n, p in para_dict.items() if p.dim() >= 2]
          nodecay_params = [p for n, p in para_dict.items() if p.dim() < 2]

          optim_groups = [
              {"params": decay_params, "weight_decay": weight_decay},
              {"params": nodecay_params, "weight_decay": 0.0},
          ]

          num_decay_params = sum(p.numel() for p in decay_params)
          num_nodecay_params = sum(p.numel() for p in nodecay_params)
          print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
          print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

          # create the AdamW optimizer using the fused kernel if available. This allows to not go through all the parameters with a loop and update them but rather use a fused kernel for efficiency
          fused_available= 'fused' in inspect.signature(torch.optim.AdamW).parameters
          use_fused = fused_available and 'cuda' in device
          print(f"using fused kernel: {use_fused}")

          optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
          return optimizer
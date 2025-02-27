"""
Multilayer Perceptron
"""
import torch.nn as nn

#FeedForward Neural Network
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.l1 = nn.Linear(config.n_embd, 4*config.n_embd) # First Linear Layer which expands the dimensionality
        self.gelu = nn.GELU(approximate="tanh") #activation function
        self.l2 = nn.Linear(4* config.n_embd, config.n_embd) # Second Linear Layer projecting back to the embedding dimension
    def forward(self, x):
        x = self.l1(x)
        x = self.gelu(x)
        x = self.l2(x)
        return x

        
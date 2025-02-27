"""
Data Loader
"""
import tiktoken
import torch

class DataLoader : # class to load our document in batches
    def __init__(self, B, T, process_rank, num_processes):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        with open('GPT2-From-Scratch/data/moliere.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"Data loaded with {len(self.tokens)} tokens") # number of tokens loaded
        print(f"1 epoch = {len(self.tokens) // (self.B * self.T)} batches") # number of batches during 1 epoch

        self.current_position = self.B * self.T * self.process_rank # setting the current position

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]
        self.current_position += B*T*self.num_processes
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        if self.current_position + (B*T*self.num_processes +1) > len(self.tokens): #if we are out of bounds, we start again at the beginning
            self.current_position = self.B * self.T * self.process_rank
        return x, y
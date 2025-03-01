"""
Data Loading Utilities
"""
import os
import tiktoken
import torch

class DataLoader:
    def __init__(self, B, T, process_rank, num_processes, data_file='data/moliere.txt'):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        # Get the project root directory
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        data_path = os.path.join(project_root, data_file)

        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Data file not found at {data_path}"
            )

        with open(data_path, 'r', encoding='utf-8') as f:
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
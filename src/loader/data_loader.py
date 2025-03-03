"""
Data Loading Utilities
"""
import os
import tiktoken
import torch
import numpy as np

# One improvement would be to shuffle the documents in the dataset when running the next epoch because the order of the documents should not matter


# load the tokens from a numpy file and return a torch tensor
def load_tokens(filename):
    npt = np.load(filename)
    ppt = torch.tensor(npt, dtype=torch.long)
    return ppt

class DataLoader:
    def __init__(self, B, T, process_rank, num_processes, split, master_process=True):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        assert split in ["train", "val"]

        # get the shard filenames
        local_path = "data/fineweb_edu"
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # Go up one more directory to reach project root
        shard_dir = os.path.join(base_path, local_path)
        shards = [f for f in os.listdir(shard_dir) if f.endswith('.npy') and split in f]
        shards = sorted(shards)
        shards = [os.path.join(shard_dir, shard) for shard in shards]
        self.shards = shards
        assert len(self.shards) > 0, f"No shards found for split {split}"

        if master_process:
            print(f"Found {len(shards)} shards for split {split}") # number of shards

        self.reset()
        
        if master_process:
            print(f"1 epoch = {len(self.tokens) // (self.B * self.T)} batches") # number of batches during 1 epoch

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]
        self.current_position += B*T*self.num_processes
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        if self.current_position + (B*T*self.num_processes +1) > len(self.tokens): #if we are out of bounds, we start at the next shards
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y
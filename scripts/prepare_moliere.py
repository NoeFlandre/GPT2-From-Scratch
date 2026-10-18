import os
import numpy as np
import tiktoken

out_dir = "data/moliere"
os.makedirs(out_dir, exist_ok=True)

filepath_train_split = os.path.join(out_dir, "moliere_train.npy")
filepath_validation_split = os.path.join(out_dir, "moliere_val.npy")

text = open("data/moliere.txt", encoding="utf8").read()
enc = tiktoken.get_encoding("gpt2")
eot = enc.encode_single_token("<|endoftext|>")
tokens = np.array([eot] + enc.encode_ordinary(text), dtype=np.uint16)

split = int(len(tokens)*0.9)
np.save(filepath_train_split, tokens[:split])
np.save(filepath_validation_split, tokens[split:])
print(f"Total tokens: {len(tokens)}")
print(f"Train tokens: {len(tokens[:split])}")
print(f"Validation tokens: {len(tokens[split:])}")

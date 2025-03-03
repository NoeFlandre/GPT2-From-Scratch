"""
This file is download the fineweb-edu dataset from huggingface/ 

https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

It is tokenizing it and saving it as shards in the data/fineweb-edu directory.

"""

import os
from datasets import load_dataset
import tiktoken
import numpy as np
import multiprocessing as mp
from tqdm import tqdm


local_path = "data/fineweb_edu"
sample_version = "sample-10BT"
shard_size = int(1e8) # 100M tokens per shard, we want 100 shards

# create the local path and the cache if it doesn't exist
DATA_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), local_path)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the dataset from huggingface
dataset = load_dataset("HuggingFaceFW/fineweb-edu", name=sample_version, split="train")

# initialize the tokenizer
tokenizer = tiktoken.get_encoding("gpt2")
eot_token = tokenizer._special_tokens['<|endoftext|>'] # end of text token

# function to tokenize a document and return it as a numpy array of uint16 tokens, these are ranging up to 65,535 which is less than our vocab size. Using uint16 is memory efficient.

def tokenize_function(doc):

    tokens = [eot_token] # start with the end of text token to delimits the beginning of the document
    tokens.extend(tokenizer.encode_ordinary(doc["text"])) # encode the document and add the tokens to the list
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary out of range for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)

    return tokens_np_uint16


# function to write a data file

def write_data_file(filename, tokens_np_uint16):
    np.save(filename, tokens_np_uint16)

# tokenize all the documents and output the shards

num_procs = max(1, os.cpu_count()//2)  # number of processes to use for tokenization

with mp.Pool(num_procs) as p:
    shard_index = 0
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16) # buffer to store the tokens
    token_count = 0
    progress_bar = None

    for tokens in p.imap(tokenize_function, dataset, chunksize=16):
        
        # we check if the current shard is enough to store the tokens
        if token_count + len(tokens) < shard_size:
            # if it is, we add the tokens to the current shard
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)

            # update the progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc = f" Shard {shard_index}")
                progress_bar.update(len(tokens))

        else :
            # write the current shard and start a new one
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"fineweb_edu_{split}_{shard_index:06d}")

            # fill the shard with what fits in it 
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_data_file(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
               
            # fill the next shard with the remaining tokens
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens) - remainder

# write the last shard if there is any with the remaining tokens
if token_count > 0:
    split = "val" if shard_index == 0 else "train"
    filename = os.path.join(DATA_CACHE_DIR, f"fineweb_edu_{split}_{shard_index:06d}")
    write_data_file(filename, all_tokens_np[:token_count])
        
















"""
HellaSwag evaluation module for commonsense reasoning evaluation. HellaSwag is based on sentence completion tasks where an LLM must select the most appropriate ending from a list of choices. 
The incorrect choices are designed to be plausible, but incorrect and easy to detect for humans but not for LLMs. Nowadays, this benchmark has been crushed by the state of the art models.
This becnhmark is interesting because it is a smooth one, meaning that the model is slowly improving on it starting from random score of 25%. The model we are training is too small to handle MCQs.
Therefore we cannot use this benchmark to evaluate the performance of the model in the way it is primarly intended. What we are rather doing is a token completion task. We build a batch of size 4 
each batch containing T tokens. Each sequence is sharing a context tokens which is the beginning of the sentence to be completed. Juxtaposed to this context, each sequence then has the following tokens 
as options: the correct answer and three distractors.

sequence = [[context, option1], [context, option2], [context, option3], [context, option4]], one of the options being the correct answer. T is defined as the maximum length of the sequence on each example.

For the rows no being of the same length than max length, we use a mask to set these padding tokens to 0. We are going for each row to compute the cross entropy loss and pick the lowest as the prediction.

Example HellaSwag json item:

{"ind": 24, "activity_label": "Roof shingle removal", "ctx_a": "A man is sitting on a roof.", "ctx_b": "he", "ctx": "A man is sitting on a roof. he", "split": "val", "split_type": "indomain", "label": 3, "endings": ["is using wrap to wrap a pair of skis.", "is ripping level tiles off.", "is holding a rubik's cube.", "starts pulling up roofing on a roof."], "source_id": "activitynet~v_-JhWjGDPHMY"}

ind: dataset ID
activity_label: The ActivityNet or WikiHow label for this example
context: There are two formats. The full context is in ctx. When the context ends in an (incomplete) noun phrase, like for ActivityNet, this incomplete noun phrase is in ctx_b, and the context up until then is in ctx_a. This can be useful for models such as BERT that need the last sentence to be complete. However, it's never required. If ctx_b is nonempty, then ctx is the same thing as ctx_a, followed by a space, then ctx_b.
endings: a list of 4 endings. The correct index is given by label (0,1,2, or 3)
split: train, val, or test.
split_type: indomain if the activity label is seen during training, else zeroshot
source_id: Which video or WikiHow article this example came from

GPT2 124M model performs acc_norm: 0.2955 (completion style)

The validation set of HellaSwag has a total of 10,042 examples.

"""

import os
import json
import requests
import tiktoken
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel

local_path = "data/hellaswag"

# create the local path and the cache if it doesn't exist
DATA_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), local_path)


hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

tokenizer = tiktoken.get_encoding("gpt2")

def download_file_url(url:str, fname:str, chunk_size:int=1024):
    """
    Download a file from a URL.
    
    """
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(fname, 'wb') as f, tqdm(
        desc=fname,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size):
            size = f.write(data)
            bar.update(size)


def load_hellaswag_data(split):
    """
    Load HellaSwag data from a JSON file.
    """
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = hellaswags[split]
    data_path = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")

    if not os.path.exists(data_path):
        print(f"Downloading {data_url} to {data_path}...")
        download_file_url(data_url, data_path)

def render_hellaswag_example(example:dict):
    """
    Render a HellaSwag example dictionary as a 3 torch.Tensor :

    -tokens (context + completion) size 4xT 
    -mask (value equals 1 for the candidate completion)
    -label (index of the correct completion)

    """

    ctx = example["ctx"]
    endings = example["endings"]
    label = example["label"]

    data = {
        "label": label,
        "ctx_tokens": None, 
        "endings_tokens": [],
    }

    ctx_tokens = tokenizer.encode(ctx)
    data["ctx_tokens"]= ctx_tokens
    token_rows = []
    mask_rows = []
    for ending in endings:
        ending_tokens = tokenizer.encode(" " + ending)
        token_rows.append(ctx_tokens + ending_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(ending_tokens))

        data["endings_tokens"].append(ending_tokens)

    max_length = max(len(row) for row in token_rows)
    tokens = torch.zeros(4, max_length, dtype=torch.long)
    mask = torch.zeros(4, max_length, dtype=torch.long)

    for i, (tokens, mask) in enumerate(zip(token_rows, mask_rows)):
        tokens[i, :len(token_rows)] = torch.tensor(token_rows)
        mask[i, :len(mask_rows)] = torch.tensor(mask_rows)
    
    return data, tokens, mask, label



    
def iterate_examples(split):
    """
    Iterate over the examples of a given split.
    """
    load_hellaswag_data(split)
    with open(os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl"), "r") as f:
         for line in f:
              example = json.loads(line)
              yield example
    
                
@torch.no_grad() # decorate the function to avoid computing the gradient during evaluation
def evaluate_mdoel(model, data_loader):
    pass
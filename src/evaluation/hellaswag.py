"""
HellaSwag evaluation module for commonsense reasoning evaluation. HellaSwag is based on sentence completion tasks where an LLM must select the most appropriate ending from a list of choices. 
The incorrect choices are designed to be plausible, but incorrect and easy to detect for humans but not for LLMs. Nowadays, this benchmark has been crushed by the state of the art models.
This becnhmark is interesting because it is a smooth one, meaning that the model is slowly improving on it starting from random score of 25%. The model we are training is too small to handle MCQs.
Therefore we cannot use this benchmark to evaluate the performance of the model in the way it is primarly intended. What we are rather doing is a token completion task. We build a batch of size 4 
each batch containing T tokens. Each sequence is sharing a context tokens which is the beginning of the sentence to be completed. Juxtaposed to this context, each sequence then has the following tokens 
as options: the correct answer and three distractors.

sequence = [[context, option1], [context, option2], [context, option3], [context, option4]], one of the options being the correct answer. T is defined as the maximum length of the sequence on each example.

For the rows not being of the same length than max length, we use a mask to set these padding tokens to 0. We are going for each row to compute the cross entropy loss and pick the lowest as the prediction.

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

Results for gpt2: 10042 acc_norm: 2967/10042=0.2955

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
DATA_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "hellaswag")


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

    for i, (tok_row, mask_row) in enumerate(zip(token_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)
    
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
def evaluate_model(model_type, device):
    torch.set_float32_matmul_precision("high") # we want to use tf32 as it speeds up the computation while keeping a good numerical accuracy
    model = GPT2LMHeadModel.from_pretrained(model_type) # we use GPT2 
    
    # Check if CUDA is available, if not use CPU
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, using CPU instead")
        device = "cpu"
    
    model.to(device)
    #model = torch.compile(model) # torch compile is creating error with the evaluation

    # Initialize counters
    num_correct = 0
    num_correct_normalized = 0
    total = 0

    for example in iterate_examples("val"):
        data, tokens, mask, label = render_hellaswag_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)
        
        #get the logits
        logits = model(tokens).logits # size 4xTXvocab
        
        #evaluate the autoregressive loss at each position
        shift_logits = logits[:, :-1, :].contiguous() #we take all the logits except the last one which is the prediction for a token outside of our window
        shift_labels = tokens[:, 1:].contiguous() #we take all the labels except the first one as it is not having any logit prediction by the model
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1)) # (4, T-1, vocab_size) -> (4(T-1), vocab_size)
        flat_shift_labels = shift_labels.view(-1) # (4, T-1) -> (4(T-1))

        #compute the cross entropy loss
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_labels, reduction="none") # we are getting a loss for each token, size (4(T-1))
        shift_losses = shift_losses.view(tokens.size(0), -1) # (4(T-1)) -> (4, T-1)

        #apply the mask to focus on the completion region

        shift_mask = (mask[..., 1:]).contiguous() # we shifted the labels so we need to shift the mask accordingly (4, T-1)
        masked_shift_losses = shift_losses * shift_mask
        sum_shift_losses = masked_shift_losses.sum(dim=1) # (4)
        avg_shift_loss = sum_shift_losses / shift_mask.sum(dim=1) # (4)
        # we pick the lowest loss out of the 4 as the most likely 
        prediction = sum_shift_losses.argmin().item()
        prediction_normalized = avg_shift_loss.argmin().item()

        #statistics
        num_correct += (prediction == label)
        num_correct_normalized += (prediction_normalized == label)
        total += 1
        print(f"{total} acc_norm: {num_correct_normalized}/{total}={num_correct_normalized/total:.4f}")

        #few examples
        if total < 10:
            print(f"Context:\n {example['ctx']}")
            print(f"Endings:")
            for i, end in enumerate(example["endings"]):
                print(f"{i} (loss: {avg_shift_loss[i].item():.4f}) {end}")
            print(f"predicted: {prediction_normalized}, actual: {label}")

if __name__ == "__main__":
     import argparse
     parser = argparse.ArgumentParser()
     parser.add_argument("-m", "--model_type", type=str, default="gpt2", help="the model type to use")
     parser.add_argument("-d", "--device", type=str, default="cuda", help="the device to use")
     args = parser.parse_args()
     evaluate_model(args.model_type, args.device)
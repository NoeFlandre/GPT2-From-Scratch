import math

max_lr = 6e-4
min_lr = max_lr*0.1
warmup_steps = 715 # 375e6 tokens (number of tokens used to warm up by GPT3 paper) / 2**19 tokens
max_steps = 19073 # 10e9 tokens / 2**19 tokens

def get_lr(it):

  # 1) Linear warmup
  if it < warmup_steps:
    return max_lr * (it + 1)/warmup_steps

  # 2) If we go beyond the maximum iterations, we set the learning rate to its minimum value which is 10%
  if it > max_steps :
    return min_lr

  # 3) In between these two phases, we are using a cosine decay
  decay_ration = (it - warmup_steps) / (max_steps - warmup_steps)
  assert 0 <= decay_ration <= 1
  cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_ration)) # from 1 to 0
  return min_lr + (max_lr - min_lr) * cosine_decay
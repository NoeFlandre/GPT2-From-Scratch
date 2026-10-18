import math


def get_lr(it, cfg):

  # 1) Linear warmup
  if it < cfg.warmup_steps:
    return cfg.max_lr * (it + 1)/cfg.warmup_steps

  # 2) If we go beyond the maximum iterations, we set the learning rate to its minimum value which is 10%
  if it > cfg.max_steps :
    return cfg.min_lr

  # 3) In between these two phases, we are using a cosine decay
  decay_ration = (it - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
  assert 0 <= decay_ration <= 1
  cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_ration)) # from 1 to 0
  return cfg.min_lr + (cfg.max_lr - cfg.min_lr) * cosine_decay

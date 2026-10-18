from dataclasses import dataclass

@dataclass
class TrainConfig():
    weight_decay: float = 0.1
    max_lr: float = 6e-4
    warmup_steps: int = 715 # 375e6 tokens (number of tokens used to warm up by GPT3 paper) / 2**19 tokens
    max_steps: int = 19073 # 10e9 tokens / 2**19 tokens
    total_batch_size: int = 524288  # 2**19
    micro_batch_size: int = 64  # micro batch size (try to fit more depending on config). To stick to GPT2 we would use 32
    sequence_length: int = 1024  # sequence length. True GPT2 is using 2048



    @property
    def min_lr(self) -> float:
        return self.max_lr*0.1

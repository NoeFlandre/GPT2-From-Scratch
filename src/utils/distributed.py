"""
Distributed Training Utilities
"""
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed():
    """Set up distributed training environment"""
    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1

    if ddp:
        print("Running in Distributed Data Parallel (DDP) mode!")
        assert torch.cuda.is_available()
        dist.init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        print("Running in single-GPU (non-DDP) mode")
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        print(f"using device: {device}")

    return {
        'ddp': ddp,
        'rank': ddp_rank,
        'local_rank': ddp_local_rank,
        'world_size': ddp_world_size,
        'device': device,
        'master_process': master_process
    }

def cleanup_distributed():
    """Clean up distributed training environment"""
    if dist.is_initialized():
        dist.destroy_process_group()

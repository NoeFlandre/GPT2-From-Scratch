"""
Main training script for GPT2
"""
import os
import sys
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# torchrun --standalone --nproc_per_node=2 train.py

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.model.gpt2 import GPT2
from src.model.config import GPT2Config
from src.loader.data_loader import DataLoader
from src.utils.distributed import setup_distributed, cleanup_distributed
from src.utils.scheduler import get_lr

def main():
    
    # Set up distributed environment
    dist_env = setup_distributed()
    device = dist_env['device']
    master_process = dist_env['master_process']

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Set matmul precision to high for better performance
    torch.set_float32_matmul_precision('high')

    # Training hyperparameters
    max_steps = 19073 # 10e9 tokens / 2**19 tokens per step
    total_batch_size = 524288  # 2**19
    micro_batch_size = 64  # micro batch size (try to fit more depending on config). To stick to GPT2 we would use 32
    sequence_length = 1024  # sequence length. True GPT2 is using 2048
    assert total_batch_size % (micro_batch_size * sequence_length * dist_env['world_size']) == 0
    grad_accum_steps = total_batch_size // (micro_batch_size * sequence_length * dist_env['world_size'])

    if master_process:
        print(f"Total desired batch size: {total_batch_size}")
        print(f"Gradient accumulation steps: {grad_accum_steps}")


    # Initialize data loader
    train_loader = DataLoader(
        B=micro_batch_size,
        T=sequence_length,
        process_rank=dist_env['rank'],
        num_processes=dist_env['world_size'],
        split="train",
        master_process=master_process
    )

    val_loader = DataLoader(
        B=micro_batch_size,
        T=sequence_length,
        process_rank=dist_env['rank'],
        num_processes=dist_env['world_size'],
        split="val",
        master_process=master_process
    )

    # Initialize model
    model = GPT2(GPT2Config(vocab_size=50304))  # Using power of 2 for efficiency
    model.to(device)
    model = torch.compile(model)  # Enable torch.compile for optimization
    
    if dist_env['ddp']:
        model = DDP(model, device_ids=[dist_env['local_rank']])
    
    raw_model = model.module if dist_env['ddp'] else model

    # Initialize optimizer
    optimizer = raw_model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=6e-4,
        device=device,
        master_process=master_process
    )

    # Training loop
    for step in range(max_steps): 
        t0 = time.time()

        # once in a while evaluate our validation loss
        if step % 100 == 0 and step > 0:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
                    
            if dist_env['ddp']:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"validation loss: {val_loss_accum.item():.4f}")

        # training loop
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0

        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x = x.to(device)
            y = y.to(device)

            # Use bfloat16 mixed precision for forward pass and loss computation
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits, loss = model(x, y)
                loss = loss / grad_accum_steps
            
            loss_accum += loss.detach()
            if dist_env['ddp']:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            loss.backward()

        if dist_env['ddp']:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        # Gradient clipping
        norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update learning rate
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Optimizer step
        optimizer.step()

        # Logging
        if master_process:
            torch.cuda.synchronize()
            t1 = time.time()
            dt = (t1 - t0) * 1000  # ms
            tokens_per_second = (train_loader.B * train_loader.T * grad_accum_steps * dist_env['world_size']) / (t1 - t0)
            print(
                f"Step {step:4d} | "
                f"Loss {loss_accum.item():.6f} | "
                f"Grad Norm {norm:.4f} | "
                f"LR {lr:.4e} | "
                f"Time {dt:.2f}ms | "
                f"Tokens/s {tokens_per_second:.2f}"
            )

    cleanup_distributed()

if __name__ == "__main__":
    main()

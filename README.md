# GPT2-From-Scratch

A repository where I try to code GPT-2 from scratch by learning from Andrej Karpathy.

## Useful Commands and Learnings

While working on this project, I’ve learned some useful commands and techniques, which I list here as a reminder for anyone including myself:

### 1. **Monitor GPU Usage**
   - To continuously monitor GPU usage every 2 seconds:
     ```bash
     watch -n 2 nvidia-smi
     ```

### 2. **SSH with Port Forwarding**
   - To SSH into a remote cluster and forward a port for accessing GPUs:
     ```bash
     ssh -p <PORT> <USER>@<REMOTE_HOST> -L <LOCAL_PORT>:localhost:<REMOTE_PORT>
     ```
   - This allows access to services running on a remote machine from your local system.

### 3. **Distributed Training with PyTorch**
   - To run a standalone multi-GPU training job:
     ```bash
     torchrun --standalone --nproc_per_node=<NUM_GPUS> train.py
     ```
   - Replace `<NUM_GPUS>` with the number of GPUs to use.

### 4. **Sync Files to Remote Server with `rsync`**
   - To transfer files from local to remote while showing progress:
     ```bash
     rsync -avz --progress -e "ssh -p <PORT>" "<LOCAL_PATH>" <USER>@<REMOTE_HOST>:<REMOTE_PATH>/
     ```
   - This ensures only modified files are transferred, saving time.

### 5. **Disable Auto-Tmux in SSH Sessions**
   - If the remote server automatically launches `tmux` and you want to disable it:
     ```bash
     touch ~/.no_auto_tmux
     ```
   - This prevents `tmux` from launching automatically, ensuring normal terminal behavior and especially get rid of synchronized terminals which can be annoying when we want to monitor.

---

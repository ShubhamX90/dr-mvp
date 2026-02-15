#!/bin/bash
export BASHRCSOURCED=1
#SBATCH -J drmvp_ddp_4gpu
#SBATCH -p gpu_h100_4
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH -c 2
#SBATCH --mem=128G
#SBATCH --time=00:15:00
#SBATCH -o runs/slurm.%j.out
#SBATCH -e runs/slurm.%j.err
set -euo pipefail
spack load /ku6kdpo
source "1248200074 1248200072 1248200074conda info --base)/etc/profile.d/conda.sh"
conda activate dr-mvp
cd $SLURM_SUBMIT_DIR
cat > runs/ddp_smoke_${SLURM_JOB_ID}.py <<'PY'
import os, torch, torch.distributed as dist
dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
t = torch.randn(2048,2048, device="cuda")
u = t @ t
torch.cuda.synchronize()
if int(os.environ.get("RANK","0")) == 0:
    print("world_size", dist.get_world_size())
    print("device0", torch.cuda.get_device_name(0))
    print("runtime", torch.version.cuda)
    print("ok", u.shape, float(u.mean()))
dist.destroy_process_group()
PY
torchrun --standalone --nproc_per_node=1 runs/ddp_smoke_${SLURM_JOB_ID}.py

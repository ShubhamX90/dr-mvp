#!/bin/bash
#SBATCH -J drmvp_env_1gpu
#SBATCH -p gpu_h100_4
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH -o runs/slurm.%j.out
#SBATCH -e runs/slurm.%j.err

set -euo pipefail
export BASHRCSOURCED=1

spack load /ku6kdpo
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dr-mvp

cd "$SLURM_SUBMIT_DIR"

python - <<'PY'
import torch
print("cuda_available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0))
print("runtime", torch.version.cuda)
x = torch.randn(4096,4096, device="cuda")
y = x @ x
torch.cuda.synchronize()
print("ok", y.shape, float(y.mean()))
PY

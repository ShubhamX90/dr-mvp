# Environment setup (Sharanga H100)

## Create env
conda create -n dr-mvp python=3.10 -y
conda activate dr-mvp
python -m pip install -U pip setuptools wheel

## Install PyTorch (CUDA 12.1 wheels)
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

## Verify GPU
python - << 'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.version.cuda)
PY

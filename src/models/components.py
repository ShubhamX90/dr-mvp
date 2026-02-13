"""Model components with GroupNorm for stable training."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GroupNormConvBlock(nn.Module):
    """Conv block with GroupNorm - works at ANY batch size."""

    def __init__(self, in_ch: int, out_ch: int, num_groups: int = 32):
        super().__init__()
        # Auto-adjust num_groups
        if out_ch < num_groups:
            num_groups = out_ch
        while out_ch % num_groups != 0:
            num_groups = num_groups // 2

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)
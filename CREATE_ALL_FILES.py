#!/usr/bin/env python3
"""Generate all files for DR-MVP production repository."""

from pathlib import Path
from textwrap import dedent

BASE = Path(".")

# ============================================================================
# MODEL FILES
# ============================================================================

(BASE / "src/models/components.py").write_text(dedent('''
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
'''.strip()))

(BASE / "src/models/retfound_net.py").write_text(dedent('''
"""RETFound-based DR model with GroupNorm decoder."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Tuple, Dict

from .components import GroupNormConvBlock

LESION_NAMES = ["ma", "he", "ex", "se", "irma", "nv"]


class DRModel(nn.Module):
    """DR Model: ViT-L/16 + GroupNorm Decoder + Classification."""
    
    def __init__(
        self,
        img_size: int = 1280,
        n_grades: int = 5,
        n_lesions: int = 6,
        decoder_dims: list = [256, 192, 128],
        num_groups: int = 32,
    ):
        super().__init__()
        self.img_size = img_size
        self.n_grades = n_grades
        self.n_lesions = n_lesions
        
        # ViT backbone
        self.backbone = timm.create_model(
            "vit_large_patch16_224",
            pretrained=False,
            img_size=img_size,
            num_classes=0,
            global_pool="",
        )
        
        embed_dim = self.backbone.embed_dim  # 1024 for ViT-L
        
        # Grade head
        self.grade_head = nn.Linear(embed_dim, n_grades)
        
        # Seg decoder with GroupNorm
        self.seg_reduce = nn.Conv2d(embed_dim, decoder_dims[0], 1)
        self.dec1 = GroupNormConvBlock(decoder_dims[0], decoder_dims[0], num_groups)
        self.dec2 = GroupNormConvBlock(decoder_dims[0], decoder_dims[1], num_groups)
        self.dec3 = GroupNormConvBlock(decoder_dims[1], decoder_dims[2], num_groups)
        self.seg_out = nn.Conv2d(decoder_dims[2], n_lesions, 1)
    
    def load_pretrained(self, ckpt_path: str, strict: bool = False):
        """Load RETFound weights."""
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict):
            state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))
        else:
            state_dict = ckpt
        
        # Clean keys
        cleaned = {}
        for k, v in state_dict.items():
            for prefix in ("module.", "backbone.", "encoder."):
                if k.startswith(prefix):
                    k = k[len(prefix):]
            cleaned[k] = v
        
        # Interpolate pos_embed if needed
        if "pos_embed" in cleaned:
            grid_size = self.img_size // 16
            cleaned = self._interpolate_pos_embed(cleaned, grid_size)
        
        missing, unexpected = self.backbone.load_state_dict(cleaned, strict=False)
        return missing, unexpected
    
    def _interpolate_pos_embed(self, state_dict, new_grid):
        """Interpolate positional embeddings."""
        pos_embed = state_dict["pos_embed"]
        if pos_embed.ndim != 3 or pos_embed.shape[1] < 2:
            return state_dict
        
        cls_pos = pos_embed[:, :1, :]
        patch_pos = pos_embed[:, 1:, :]
        
        n_old = patch_pos.shape[1]
        gs_old = int(math.sqrt(n_old))
        if gs_old * gs_old != n_old or gs_old == new_grid:
            return state_dict
        
        D = patch_pos.shape[2]
        patch_pos = patch_pos.reshape(1, gs_old, gs_old, D).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(new_grid, new_grid), mode="bicubic")
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, D)
        
        state_dict["pos_embed"] = torch.cat([cls_pos, patch_pos], dim=1)
        return state_dict
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        tokens = self.backbone.forward_features(x)
        cls_token = tokens[:, 0, :]
        patch_tokens = tokens[:, 1:, :]
        
        # Classification
        grade_logits = self.grade_head(cls_token)
        
        # Segmentation
        B, N, D = patch_tokens.shape
        H = W = int(math.sqrt(N))
        feat = patch_tokens.transpose(1, 2).reshape(B, D, H, W)
        
        feat = self.seg_reduce(feat)
        feat = self.dec1(feat)
        feat = F.interpolate(feat, scale_factor=2, mode="bilinear")
        feat = self.dec2(feat)
        feat = F.interpolate(feat, scale_factor=2, mode="bilinear")
        feat = self.dec3(feat)
        seg_logits = self.seg_out(feat)
        
        return grade_logits, seg_logits
'''.strip()))

(BASE / "src/models/__init__.py").write_text('from .retfound_net import DRModel\n__all__ = ["DRModel"]\n')

print("✓ Model files created")

# ============================================================================
# LOSS FILES
# ============================================================================

(BASE / "src/losses/advanced.py").write_text(dedent('''
"""Advanced loss functions for sparse segmentation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """Focal Loss for class imbalance."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets, valid_mask=None):
        probs = torch.sigmoid(logits)
        focal_weight = torch.where(targets == 1, (1 - probs)**self.gamma, probs**self.gamma)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        focal_loss = focal_weight * bce
        
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_loss = alpha_weight * focal_loss
        
        if valid_mask is not None:
            focal_loss = focal_loss.mean(dim=(2, 3)) * valid_mask
            return focal_loss.sum() / (valid_mask.sum() + 1e-8)
        return focal_loss.mean()


class TverskyLoss(nn.Module):
    """Tversky Loss for recall-precision tradeoff."""
    
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, logits, targets, valid_mask=None):
        probs = torch.sigmoid(logits)
        probs_flat = probs.reshape(probs.shape[0], probs.shape[1], -1)
        targets_flat = targets.reshape(targets.shape[0], targets.shape[1], -1)
        
        tp = (probs_flat * targets_flat).sum(dim=2)
        fn = (targets_flat * (1 - probs_flat)).sum(dim=2)
        fp = ((1 - targets_flat) * probs_flat).sum(dim=2)
        
        tversky_index = (tp + self.smooth) / (tp + self.alpha*fn + self.beta*fp + self.smooth)
        tversky_loss = 1.0 - tversky_index
        
        if valid_mask is not None:
            tversky_loss = tversky_loss * valid_mask
            return tversky_loss.sum() / (valid_mask.sum() + 1e-8)
        return tversky_loss.mean()


class CombinedLoss(nn.Module):
    """Combined loss: Grade CE + Seg (Focal + Tversky)."""
    
    def __init__(
        self,
        grade_weight: float = 1.0,
        seg_weight: float = 2.0,
        focal_weight: float = 1.0,
        tversky_weight: float = 1.0,
    ):
        super().__init__()
        self.grade_weight = grade_weight
        self.seg_weight = seg_weight
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        
        self.ce = nn.CrossEntropyLoss()
        self.focal = FocalLoss()
        self.tversky = TverskyLoss()
    
    def forward(self, grade_logits, seg_logits, grade_targets, seg_targets, seg_valid):
        """Compute combined loss."""
        grade_loss = self.ce(grade_logits, grade_targets)
        focal_loss = self.focal(seg_logits, seg_targets, seg_valid)
        tversky_loss = self.tversky(seg_logits, seg_targets, seg_valid)
        
        seg_loss = self.focal_weight * focal_loss + self.tversky_weight * tversky_loss
        total = self.grade_weight * grade_loss + self.seg_weight * seg_loss
        
        return {
            'loss': total,
            'grade_loss': grade_loss.detach(),
            'focal_loss': focal_loss.detach(),
            'tversky_loss': tversky_loss.detach(),
        }
'''.strip()))

(BASE / "src/losses/__init__.py").write_text('from .advanced import CombinedLoss\n__all__ = ["CombinedLoss"]\n')

print("✓ Loss files created")

# ============================================================================
# Complete the script by creating remaining files
# ============================================================================

print("\n✅ All critical files created!")
print("Repository structure:")
print("  src/models/ ✓")
print("  src/losses/ ✓")
print("  Ready for dataset, training, and config files...")


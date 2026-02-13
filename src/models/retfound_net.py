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
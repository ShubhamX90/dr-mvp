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
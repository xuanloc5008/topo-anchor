from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def one_hot_targets(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    if target.ndim >= 5 and target.shape[1] == 1:
        target = target[:, 0]
    target = target.long()
    return F.one_hot(target, num_classes=num_classes).movedim(-1, 1).float()


class MulticlassDiceLoss(nn.Module):
    def __init__(self, *, include_background: bool = False, smooth: float = 1e-5) -> None:
        super().__init__()
        self.include_background = include_background
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        prob = torch.softmax(logits, dim=1)
        target_one_hot = one_hot_targets(target, num_classes=num_classes).to(device=logits.device)
        if not self.include_background and num_classes > 1:
            prob = prob[:, 1:]
            target_one_hot = target_one_hot[:, 1:]
        dims = tuple(range(2, prob.ndim))
        intersection = torch.sum(prob * target_one_hot, dim=dims)
        denominator = torch.sum(prob + target_one_hot, dim=dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class DiceCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        *,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        include_background: bool = False,
    ) -> None:
        super().__init__()
        self.dice_weight = float(dice_weight)
        self.ce_weight = float(ce_weight)
        self.dice = MulticlassDiceLoss(include_background=include_background)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim >= 5 and target.shape[1] == 1:
            target = target[:, 0]
        target = target.long()
        return self.dice_weight * self.dice(logits, target) + self.ce_weight * self.ce(logits, target)

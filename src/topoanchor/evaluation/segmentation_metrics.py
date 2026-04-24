from __future__ import annotations

import numpy as np
import torch


def multiclass_dice_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    include_background: bool = False,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    pred = torch.argmax(logits, dim=1)
    if target.ndim >= 5 and target.shape[1] == 1:
        target = target[:, 0]
    num_classes = logits.shape[1]
    class_range = range(0 if include_background else 1, num_classes)
    scores = []
    for class_id in class_range:
        pred_c = pred == class_id
        target_c = target == class_id
        intersection = (pred_c & target_c).sum().float()
        denominator = pred_c.sum().float() + target_c.sum().float()
        scores.append((2.0 * intersection + epsilon) / (denominator + epsilon))
    if not scores:
        return torch.tensor(1.0, device=logits.device)
    return torch.stack(scores).mean()


def dice_iou_per_class(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    num_classes: int,
    include_background: bool = False,
    epsilon: float = 1e-6,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    class_range = range(0 if include_background else 1, num_classes)
    dice_values = []
    iou_values = []
    for class_id in class_range:
        pred_c = pred == class_id
        target_c = target == class_id
        intersection = float(np.logical_and(pred_c, target_c).sum())
        pred_sum = float(pred_c.sum())
        target_sum = float(target_c.sum())
        union = float(np.logical_or(pred_c, target_c).sum())
        dice = (2.0 * intersection + epsilon) / (pred_sum + target_sum + epsilon)
        iou = (intersection + epsilon) / (union + epsilon)
        metrics[f"dice_class_{class_id}"] = dice
        metrics[f"iou_class_{class_id}"] = iou
        dice_values.append(dice)
        iou_values.append(iou)
    metrics["dice_mean"] = float(np.mean(dice_values)) if dice_values else 1.0
    metrics["iou_mean"] = float(np.mean(iou_values)) if iou_values else 1.0
    return metrics

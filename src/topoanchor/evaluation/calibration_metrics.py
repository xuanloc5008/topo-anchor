from __future__ import annotations

import numpy as np
import torch


def raw_segmentation_confidence(prob: torch.Tensor) -> torch.Tensor:
    return prob.max(dim=1).values.flatten(1).mean(dim=1)


def calibrated_confidence(raw_confidence: torch.Tensor, mahalanobis: torch.Tensor, *, gamma: float) -> torch.Tensor:
    return raw_confidence * torch.exp(-float(gamma) * mahalanobis)


def expected_calibration_error(
    confidences: np.ndarray,
    correct: np.ndarray,
    *,
    num_bins: int = 15,
) -> float:
    confidences = np.asarray(confidences, dtype=np.float64).reshape(-1)
    correct = np.asarray(correct, dtype=np.float64).reshape(-1)
    if confidences.size != correct.size:
        raise ValueError("confidences and correct must have the same number of elements.")
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (confidences > left) & (confidences <= right)
        if not mask.any():
            continue
        accuracy = correct[mask].mean()
        confidence = confidences[mask].mean()
        ece += (mask.mean()) * abs(accuracy - confidence)
    return float(ece)


def brier_score(probabilities: np.ndarray, labels: np.ndarray, *, num_classes: int) -> float:
    labels = labels.astype(np.int64)
    one_hot = np.eye(num_classes, dtype=np.float64)[labels.reshape(-1)]
    probs = probabilities.reshape(num_classes, -1).T.astype(np.float64)
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def voxel_confidence_correctness(probabilities: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels).astype(np.int64)
    pred = probabilities.argmax(axis=0)
    confidence = probabilities.max(axis=0)
    correct = pred == labels
    return confidence.reshape(-1), correct.reshape(-1)

from __future__ import annotations

import numpy as np

from topoanchor.utils.imports import require_package


def _surface(mask: np.ndarray) -> np.ndarray:
    ndimage = require_package("scipy.ndimage", "pip install scipy")
    mask = mask.astype(bool)
    if not mask.any():
        return mask
    eroded = ndimage.binary_erosion(mask)
    return mask ^ eroded


def hausdorff_assd_binary(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    ndimage = require_package("scipy.ndimage", "pip install scipy")
    pred_surface = _surface(pred)
    target_surface = _surface(target)
    if not pred_surface.any() and not target_surface.any():
        return 0.0, 0.0
    if not pred_surface.any() or not target_surface.any():
        return float("inf"), float("inf")

    target_distance = ndimage.distance_transform_edt(~target_surface)
    pred_distance = ndimage.distance_transform_edt(~pred_surface)
    pred_to_target = target_distance[pred_surface]
    target_to_pred = pred_distance[target_surface]
    hd = max(float(pred_to_target.max()), float(target_to_pred.max()))
    assd = float((pred_to_target.mean() + target_to_pred.mean()) / 2.0)
    return hd, assd


def surface_metrics_per_class(pred: np.ndarray, target: np.ndarray, *, num_classes: int) -> dict[str, float]:
    out: dict[str, float] = {}
    for class_id in range(1, num_classes):
        hd, assd = hausdorff_assd_binary(pred == class_id, target == class_id)
        out[f"hausdorff_class_{class_id}"] = hd
        out[f"assd_class_{class_id}"] = assd
    finite_hd = [value for key, value in out.items() if key.startswith("hausdorff") and np.isfinite(value)]
    finite_assd = [value for key, value in out.items() if key.startswith("assd") and np.isfinite(value)]
    out["hausdorff_mean_finite"] = float(np.mean(finite_hd)) if finite_hd else float("inf")
    out["assd_mean_finite"] = float(np.mean(finite_assd)) if finite_assd else float("inf")
    return out

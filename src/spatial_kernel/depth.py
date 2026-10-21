from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def load_depth_map(path: str, target_shape: tuple[int, int] | None = None) -> np.ndarray:
    if path.lower().endswith(".npy"):
        depth = np.load(path)
    else:
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"Could not decode depth map: {path}")
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    depth = depth.astype(np.float32)
    if target_shape is not None and depth.shape != target_shape:
        h, w = target_shape
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
    return depth


def compute_depth_metrics(
    depth: np.ndarray,
    mass_mask: np.ndarray,
    table_z: float | None = None,
    min_height: float = 1e-3,
) -> dict[str, Any]:
    valid = np.isfinite(depth)
    if not np.any(valid):
        return {"valid": 0, "quality_note": "no finite depth samples"}

    if table_z is None:
        table_z = float(np.nanpercentile(depth[valid], 5.0))

    height = depth - table_z
    occupied = valid & (height > min_height)
    mass = (mass_mask > 0) & valid
    weighted = occupied | mass

    ys, xs = np.nonzero(weighted)
    if len(xs) == 0:
        return {
            "valid": 1,
            "table_z": table_z,
            "delta_z": 0.0,
            "z_span": 0.0,
            "height_p95": 0.0,
            "volume_density": 0.0,
            "clearance": float(np.nanmax(height[valid])) if np.any(valid) else 0.0,
        }

    h, w = depth.shape
    z_vals = height[weighted]
    z_mean = float(np.nanmean(z_vals))
    z_p05 = float(np.nanpercentile(z_vals, 5.0))
    z_p95 = float(np.nanpercentile(z_vals, 95.0))
    z_span = float(z_p95 - z_p05)
    occupied_ratio = float(occupied.sum() / float(h * w))
    max_height = float(np.nanmax(height[valid]))
    return {
        "valid": 1,
        "table_z": float(table_z),
        "delta_z": z_mean,
        "z_span": z_span,
        "height_p95": z_p95,
        "volume_density": occupied_ratio,
        "clearance": max(0.0, max_height - z_p95),
    }

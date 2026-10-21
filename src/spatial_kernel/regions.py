from __future__ import annotations

import numpy as np


def region_mask(shape: tuple[int, int], region: str) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    region = region.lower()
    if region == "full":
        mask[:, :] = True
    elif region == "bottom_left":
        mask[h // 2 :, : w // 2] = True
    elif region == "bottom_right":
        mask[h // 2 :, w // 2 :] = True
    elif region == "top_left":
        mask[: h // 2, : w // 2] = True
    elif region == "top_right":
        mask[: h // 2, w // 2 :] = True
    elif region == "center":
        x0 = int(w * 0.25)
        x1 = int(w * 0.75)
        y0 = int(h * 0.25)
        y1 = int(h * 0.75)
        mask[y0:y1, x0:x1] = True
    else:
        raise ValueError(f"Unknown region: {region}")
    return mask


def region_center_normalized(region: str) -> tuple[float, float]:
    centers = {
        "full": (0.0, 0.0),
        "center": (0.0, 0.0),
        "bottom_left": (-0.25, 0.25),
        "bottom_right": (0.25, 0.25),
        "top_left": (-0.25, -0.25),
        "top_right": (0.25, -0.25),
    }
    if region not in centers:
        raise ValueError(f"Unknown region: {region}")
    return centers[region]


def mass_centroid_px(mass_mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mass_mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def normalized_to_px(nx: float, ny: float, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    x = int(round((nx + 0.5) * w))
    y = int(round((ny + 0.5) * h))
    return max(0, min(w - 1, x)), max(0, min(h - 1, y))


def region_metrics(mass_mask: np.ndarray, target_region: str) -> dict[str, float]:
    mass = mass_mask > 0
    total_mass = float(mass.sum())
    target = region_mask(mass.shape, target_region)
    target_mass = float((mass & target).sum())
    target_area = float(target.sum())
    outside_mass = max(0.0, total_mass - target_mass)
    frame_area = float(mass.size)
    if total_mass == 0:
        return {
            "target_mass_ratio": 0.0,
            "leakage_ratio": 0.0,
            "target_region_density": 0.0,
            "target_region_void": 1.0,
            "outside_mass_fraction": 0.0,
        }
    return {
        "target_mass_ratio": float(target_mass / total_mass),
        "leakage_ratio": float(outside_mass / total_mass),
        "target_region_density": float(target_mass / max(target_area, 1.0)),
        "target_region_void": float(1.0 - target_mass / max(target_area, 1.0)),
        "outside_mass_fraction": float(outside_mass / frame_area),
    }

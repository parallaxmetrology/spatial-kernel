from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import ConvexHull
from skimage.measure import label
from skimage.morphology import skeletonize


EPS = 1e-9


@dataclass(frozen=True)
class KernelConfig:
    target_max_side: int = 1536
    grad_low_pct: float = 85.0
    grad_high_pct: float = 97.0
    edge_margin_px: int = 2
    min_mass_frac: float = 0.001
    warn_mass_frac: float = 0.03
    r_v_absolute_threshold: float = 0.15
    orient_bins: int = 8


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_image_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def decode_image_bgr(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes")
    return img


def resize_max_side(img_bgr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    scale = max_side / float(max(h, w))
    if scale >= 1.0:
        return img_bgr.copy()
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def crop_roi(img_bgr: np.ndarray, roi: tuple[int, int, int, int] | None) -> tuple[np.ndarray, dict[str, Any]]:
    if roi is None:
        h, w = img_bgr.shape[:2]
        return img_bgr, {"x": 0, "y": 0, "w": w, "h": h, "applied": False}
    x, y, w, h = roi
    ih, iw = img_bgr.shape[:2]
    x0 = max(0, min(iw - 1, x))
    y0 = max(0, min(ih - 1, y))
    x1 = max(x0 + 1, min(iw, x0 + w))
    y1 = max(y0 + 1, min(ih, y0 + h))
    cropped = img_bgr[y0:y1, x0:x1].copy()
    return cropped, {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0, "applied": True}


def standardize_image_from_path(
    path: str,
    config: KernelConfig = KernelConfig(),
    roi: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    image_bytes = load_image_bytes(path)
    decoded = decode_image_bgr(image_bytes)
    cropped, roi_meta = crop_roi(decoded, roi)
    standardized = resize_max_side(cropped, config.target_max_side)
    h, w = standardized.shape[:2]
    meta = {
        "path": path,
        "input_sha256": sha256_bytes(image_bytes),
        "original_shape": list(decoded.shape),
        "roi": roi_meta,
        "standardized_shape": [h, w, int(standardized.shape[2])],
        "target_max_side": config.target_max_side,
    }
    return standardized, meta


def bgr_to_gray_float(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0


def sobel_gradients(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.sqrt(gx * gx + gy * gy)
    return gx, gy, gmag


def robust_threshold_mask(gmag: np.ndarray, config: KernelConfig = KernelConfig()) -> np.ndarray:
    h, w = gmag.shape
    margin = config.edge_margin_px
    inner = gmag[margin : h - margin, margin : w - margin] if h > 2 * margin and w > 2 * margin else gmag
    flat = inner.reshape(-1)
    t_low = np.percentile(flat, config.grad_low_pct)
    t_high = np.percentile(flat, config.grad_high_pct)
    if t_high <= t_low:
        mask = gmag > t_low
    else:
        mask = (gmag > t_low) & (gmag <= t_high)
    if margin > 0 and h > 2 * margin and w > 2 * margin:
        mask[:margin, :] = False
        mask[-margin:, :] = False
        mask[:, :margin] = False
        mask[:, -margin:] = False
    return mask.astype(np.uint8)


def metric_delta_x(mass_mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mass_mask)
    if len(xs) == 0:
        return 0.0
    h, w = mass_mask.shape
    return float((xs.mean() - ((w - 1) / 2.0)) / float(w))


def metric_delta_y(mass_mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mass_mask)
    if len(ys) == 0:
        return 0.0
    h, w = mass_mask.shape
    return float((ys.mean() - ((h - 1) / 2.0)) / float(h))


def metric_mass_fraction(mass_mask: np.ndarray) -> float:
    h, w = mass_mask.shape
    return float(mass_mask.sum() / float(h * w))


def metric_r_v(gmag: np.ndarray, config: KernelConfig = KernelConfig()) -> float:
    coverage = float((gmag >= config.r_v_absolute_threshold).sum() / gmag.size)
    return float(1.0 - coverage)


def metric_rho_r(mass_mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mass_mask)
    if len(xs) < 3:
        return 0.0
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 0.0
    return float(100.0 * (len(xs) / (float(hull.volume) + EPS)))


def metric_mu(mass_mask: np.ndarray) -> float:
    lab = label(mass_mask > 0, connectivity=2)
    num_components = int(lab.max())
    if num_components == 0:
        return 0.0
    if num_components == 1:
        return 1.0
    areas = np.array([int(np.sum(lab == i)) for i in range(1, num_components + 1)], dtype=np.float64)
    probs = areas / (areas.sum() + EPS)
    p_max = float(probs.max())
    entropy = float(-np.sum([p * np.log2(p) for p in probs if p > 0]))
    max_entropy = float(np.log2(num_components))
    if max_entropy == 0:
        return p_max
    return float(p_max * (1.0 - entropy / max_entropy))


def metric_x_p(gradient_mag: np.ndarray) -> float:
    h, w = gradient_mag.shape
    edge_width = 0.15
    x_left = int(w * edge_width)
    x_right = int(w * (1.0 - edge_width))
    y_top = int(h * edge_width)
    y_bottom = int(h * (1.0 - edge_width))
    edge_mask = np.zeros((h, w), dtype=bool)
    edge_mask[:y_top, :] = True
    edge_mask[y_bottom:, :] = True
    edge_mask[:, :x_left] = True
    edge_mask[:, x_right:] = True
    total = float(gradient_mag.sum())
    if total == 0:
        return 0.0
    return float(gradient_mag[edge_mask].sum() / total)


def metric_theta(
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    gradient_mag: np.ndarray,
    mass_mask: np.ndarray,
    config: KernelConfig = KernelConfig(),
) -> float:
    if config.orient_bins != 8:
        raise ValueError("Canonical orientation bin count is 8")
    mask_bool = mass_mask > 0
    gx = gradient_x[mask_bool]
    gy = gradient_y[mask_bool]
    weights = gradient_mag[mask_bool]
    if len(gx) == 0 or float(weights.sum()) == 0:
        return 0.0
    orientations = np.arctan2(gy, gx) + np.pi
    bin_width = 2.0 * np.pi / config.orient_bins
    bins = np.clip(np.floor(orientations / bin_width).astype(int), 0, config.orient_bins - 1)
    bin_weights = np.zeros(config.orient_bins, dtype=np.float64)
    np.add.at(bin_weights, bins, weights)
    probs = bin_weights / (bin_weights.sum() + EPS)
    entropy = float(-np.sum([p * np.log2(p) for p in probs if p > 0]))
    max_entropy = float(np.log2(config.orient_bins))
    return float(1.0 - entropy / max_entropy)


def metric_d_s(mass_mask: np.ndarray) -> float:
    h, w = mass_mask.shape
    if mass_mask.sum() == 0:
        return 0.0
    mask = mass_mask > 0
    dt = distance_transform_edt(mask)
    sk = skeletonize(mask).astype(np.uint8)
    ys, xs = np.nonzero(sk)
    if len(xs) == 0:
        return 0.0
    thickness = 2.0 * dt[ys, xs]
    return float(thickness.mean() / (float(min(h, w)) + EPS))


def metric_sdi(mass_mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mass_mask)
    if len(xs) == 0:
        return 0.0
    cx = xs.mean()
    cy = ys.mean()
    distances = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    h, w = mass_mask.shape
    return float(distances.mean() / (np.sqrt(h**2 + w**2) + EPS))


def mask_sha256(mass_mask: np.ndarray) -> str:
    mb = np.ascontiguousarray((mass_mask > 0).astype(np.uint8))
    return hashlib.sha256(mb.tobytes()).hexdigest()


def compute_kernel_metrics_from_bgr(
    img_bgr: np.ndarray,
    config: KernelConfig = KernelConfig(),
    object_mask: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    gray = bgr_to_gray_float(img_bgr)
    gx, gy, gmag = sobel_gradients(gray)
    mass_mask = robust_threshold_mask(gmag, config)
    if object_mask is not None:
        # Restrict structural mass to the isolated manipuland. Whole-frame fields
        # (gmag, r_v) remain scene-level; centroid/packing/dispersion become
        # object-scoped instead of measuring scene texture.
        obj = np.asarray(object_mask) > 0
        restricted = mass_mask.astype(bool) & obj
        # A flat / low-texture object can have its hard border excluded by the
        # 85-97 percentile band, leaving the intersection empty. We have direct
        # evidence those pixels ARE the object (depth/segmenter), so fall back to
        # the silhouette and measure the object's geometry rather than nothing.
        if int(restricted.sum()) < max(1, int(config.min_mass_frac * obj.size)) and int(obj.sum()) > 0:
            mass_mask = obj.astype(np.uint8)
        else:
            mass_mask = restricted.astype(np.uint8)
    mass_frac = metric_mass_fraction(mass_mask)
    fields = {"gray": gray, "gradient_x": gx, "gradient_y": gy, "gmag": gmag, "mass_mask": mass_mask}
    metrics: dict[str, Any] = {
        "delta_x": 0.0,
        "delta_y": 0.0,
        "r_v": metric_r_v(gmag, config),
        "rho_r": 0.0,
        "mu": 0.0,
        "x_p": 0.0,
        "theta": 0.0,
        "d_s": 0.0,
        "sdi": 0.0,
        "mass_fraction": float(mass_frac),
        "gradient_floor_85": float(np.percentile(gmag.flatten(), config.grad_low_pct)),
        "gradient_ceiling_97": float(np.percentile(gmag.flatten(), config.grad_high_pct)),
        "r_v_threshold": config.r_v_absolute_threshold,
        "mask_sha256": mask_sha256(mass_mask),
        "valid": 1,
        "quality_note": "",
    }
    if mass_frac < config.min_mass_frac:
        metrics["valid"] = 0
        metrics["quality_note"] = f"mass_fraction={mass_frac:.6f} < min_mass_frac={config.min_mass_frac:.6f}"
        return metrics, fields
    metrics.update(
        {
            "delta_x": metric_delta_x(mass_mask),
            "delta_y": metric_delta_y(mass_mask),
            "rho_r": metric_rho_r(mass_mask),
            "mu": metric_mu(mass_mask),
            "x_p": metric_x_p(gmag),
            "theta": metric_theta(gx, gy, gmag, mass_mask, config),
            "d_s": metric_d_s(mass_mask),
            "sdi": metric_sdi(mass_mask),
        }
    )
    if mass_frac < config.warn_mass_frac:
        metrics["quality_note"] = (
            f"mass_fraction={mass_frac:.4f} in low-structure band "
            f"({config.min_mass_frac:.3f}-{config.warn_mass_frac:.3f}); treat as low-confidence."
        )
    return metrics, fields


def compute_kernel_metrics_from_path(
    path: str,
    config: KernelConfig = KernelConfig(),
    roi: tuple[int, int, int, int] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    img_bgr, meta = standardize_image_from_path(path, config, roi)
    metrics, fields = compute_kernel_metrics_from_bgr(img_bgr, config)
    return metrics, fields, meta

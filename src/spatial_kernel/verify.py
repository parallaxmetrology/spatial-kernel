from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from .core import (
    KernelConfig,
    compute_kernel_metrics_from_bgr,
    standardize_image_from_path,
)
from .depth import compute_depth_metrics, load_depth_map
from .isolate import isolate_object
from .regions import mass_centroid_px, normalized_to_px, region_center_normalized, region_metrics
from .tasks import TaskProfile, get_task_profile


def weighted_gap(metrics: dict[str, Any], profile: TaskProfile) -> tuple[float, dict[str, float]]:
    gaps: dict[str, float] = {}
    total = 0.0
    for key, target_value in profile.target.items():
        measured = float(metrics.get(key, 0.0))
        gap = measured - float(target_value)
        gaps[key] = gap
        weight = float(profile.weights.get(key, 1.0))
        total += (gap * weight) ** 2
    return float(math.sqrt(total)), gaps


def failure_modes(
    metrics: dict[str, Any],
    region: dict[str, float],
    gaps: dict[str, float],
    profile: TaskProfile,
) -> list[str]:
    modes: list[str] = []
    if abs(gaps.get("delta_x", 0.0)) > 0.15 or abs(gaps.get("delta_y", 0.0)) > 0.15:
        modes.append("centroid_miss")
    if profile.min_target_mass_ratio is not None and region["target_mass_ratio"] < profile.min_target_mass_ratio:
        modes.append("target_under_occupied")
    if profile.max_leakage_ratio is not None and region["leakage_ratio"] > profile.max_leakage_ratio:
        modes.append("outside_leakage")
    if gaps.get("r_v", 0.0) > 0.12:
        modes.append("under_packed_or_too_void")
    if gaps.get("sdi", 0.0) > 0.10:
        modes.append("over_dispersed")
    if gaps.get("x_p", 0.0) > 0.15:
        modes.append("edge_loaded")
    if not modes:
        modes.append("within_task_tolerance")
    return modes


def correction_vector(metrics: dict[str, Any], profile: TaskProfile) -> dict[str, Any]:
    target_dx = float(profile.target.get("delta_x", region_center_normalized(profile.target_region)[0]))
    target_dy = float(profile.target.get("delta_y", region_center_normalized(profile.target_region)[1]))
    measured_dx = float(metrics.get("delta_x", 0.0))
    measured_dy = float(metrics.get("delta_y", 0.0))
    move_x = target_dx - measured_dx
    move_y = target_dy - measured_dy
    direction = []
    if abs(move_x) > 0.03:
        direction.append("left" if move_x < 0 else "right")
    if abs(move_y) > 0.03:
        direction.append("up" if move_y < 0 else "down")
    return {
        "delta_x": move_x,
        "delta_y": move_y,
        "field_direction": "/".join(direction) if direction else "none",
    }


def verify_spatial_task(
    image_path: str,
    task: str | TaskProfile = "bottom_left",
    roi: tuple[int, int, int, int] | None = None,
    depth_path: str | None = None,
    table_z: float | None = None,
    config: KernelConfig = KernelConfig(),
    isolate: str = "none",
    object_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Verify a spatial task.

    ``isolate`` controls object isolation before the kernel runs:
      "none" (default)  — legacy whole-frame gradient mass (back-compatible).
      "auto"            — explicit mask > depth plane > largest component.
      "depth_plane" | "largest_component" | "explicit" — force one strategy.
    Isolation turns ``delta_x/delta_y`` from "centroid of scene texture" into
    "centroid of the manipuland", which is the difference between a verifier that
    works in clutter and one that fails silently.
    """
    profile = get_task_profile(task) if isinstance(task, str) else task

    img_bgr, meta = standardize_image_from_path(image_path, config=config, roi=roi)
    standardized_shape = img_bgr.shape[:2]

    depth = None
    if depth_path is not None:
        depth = load_depth_map(depth_path, target_shape=standardized_shape)

    # First pass: whole-frame gradient mass (also the source for isolation).
    metrics, fields = compute_kernel_metrics_from_bgr(img_bgr, config=config)
    isolation_report: dict[str, Any] = {"strategy": "none", "applied": False}
    if isolate != "none":
        iso = isolate_object(
            fields["mass_mask"],
            object_mask=object_mask,
            depth=depth,
            table_z=table_z,
            strategy=isolate,
        )
        # Re-measure on the isolated object.
        metrics, fields = compute_kernel_metrics_from_bgr(
            img_bgr, config=config, object_mask=iso.object_mask
        )
        isolation_report = {
            "applied": True,
            "strategy": iso.strategy,
            "confidence": iso.confidence,
            "kept_mass_fraction": round(iso.kept_mass_fraction, 4),
            "notes": iso.notes,
            "provenance": iso.provenance,
        }

    region = region_metrics(fields["mass_mask"], profile.target_region)
    score, gaps = weighted_gap(metrics, profile)

    region_fail = False
    if profile.min_target_mass_ratio is not None:
        region_fail = region_fail or region["target_mass_ratio"] < profile.min_target_mass_ratio
    if profile.max_leakage_ratio is not None:
        region_fail = region_fail or region["leakage_ratio"] > profile.max_leakage_ratio

    status = "FAIL" if score > profile.threshold or region_fail or int(metrics.get("valid", 0)) == 0 else "PASS"
    centroid = mass_centroid_px(fields["mass_mask"])
    target_px = normalized_to_px(
        float(profile.target.get("delta_x", region_center_normalized(profile.target_region)[0])),
        float(profile.target.get("delta_y", region_center_normalized(profile.target_region)[1])),
        fields["mass_mask"].shape,
    )

    report: dict[str, Any] = {
        "schema_version": "vtl-spatial-compliance/v1",
        "isolation": isolation_report,
        "image": meta,
        "task": {
            "name": profile.name,
            "target_region": profile.target_region,
            "target": profile.target,
            "weights": profile.weights,
            "threshold": profile.threshold,
            "min_target_mass_ratio": profile.min_target_mass_ratio,
            "max_leakage_ratio": profile.max_leakage_ratio,
            "notes": profile.notes,
        },
        "visual_field": metrics,
        "region": region,
        "compliance": {
            "status": status,
            "score": score,
            "gaps": gaps,
            "largest_gap": max(gaps, key=lambda k: abs(gaps[k])) if gaps else None,
            "failure_modes": failure_modes(metrics, region, gaps, profile),
            "correction": correction_vector(metrics, profile),
            "measured_centroid_px": list(centroid) if centroid else None,
            "target_px": list(target_px),
        },
    }

    if depth is not None:
        report["depth_field"] = compute_depth_metrics(depth, fields["mass_mask"], table_z=table_z)

    return report


def _centroid_in_region(centroid_norm: tuple[float, float], region: str, shape: tuple[int, int]) -> bool:
    from .regions import region_mask

    px, py = normalized_to_px(centroid_norm[0], centroid_norm[1], shape)
    rmask = region_mask(shape, region)
    return bool(rmask[py, px])


def _depth_occupancy_read(
    depth: np.ndarray,
    table_z: float | None,
    region: str,
    min_height: float = 1e-3,
):
    """An RGB-independent read: object position from geometric occupancy alone.

    The occupancy threshold is noise-aware: a fixed tiny threshold floods the frame
    with phantom occupancy under any real sensor noise. We estimate the table-plane
    noise (robust MAD) and require height above ~4σ, so moderate noise is rejected
    while genuine above-plane mass survives.
    """
    from .parallax import Read

    valid = np.isfinite(depth)
    tz = float(np.nanpercentile(depth[valid], 5.0)) if table_z is None else table_z
    resid = depth[valid] - tz
    near_table = resid[resid < np.nanpercentile(resid, 60.0)]  # table pixels only
    if near_table.size > 0:
        mad = float(np.nanmedian(np.abs(near_table - np.nanmedian(near_table))))
        sigma = 1.4826 * mad
    else:
        sigma = 0.0
    thresh = max(min_height, 4.0 * sigma)
    occ = valid & ((depth - tz) > thresh)
    ys, xs = np.nonzero(occ)
    h, w = depth.shape
    if len(xs) == 0:
        return Read("depth_occupancy", "depth", (0.0, 0.0), False, note="no object above plane")
    nx = float((xs.mean() - (w - 1) / 2.0) / w)
    ny = float((ys.mean() - (h - 1) / 2.0) / h)
    in_region = _centroid_in_region((nx, ny), region, (h, w))
    return Read("depth_occupancy", "depth", (nx, ny), in_region, note=f"table_z={tz:.3f}")


def verify_parallax(
    image_path: str,
    task: str | TaskProfile = "bottom_left",
    depth_path: str | None = None,
    table_z: float | None = None,
    roi: tuple[int, int, int, int] | None = None,
    config: KernelConfig = KernelConfig(),
    rgb_isolate: str = "largest_component",
    object_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Success as the agreement of independent reads, with Ω as confidence.

    Runs an RGB structural read and (if depth is supplied) an RGB-independent depth
    occupancy read, then fuses them: success requires they agree the object is in
    the target region *and* agree on where it is (low Ω). The disagreement Ω is
    reported as the honest uncertainty — a single coherent read is never trusted
    blindly. This is the package's actual thesis made operational.
    """
    from .parallax import Read, fuse_reads

    profile = get_task_profile(task) if isinstance(task, str) else task
    base = verify_spatial_task(
        image_path, task=profile, roi=roi, config=config,
        isolate=rgb_isolate, object_mask=object_mask,
    )
    shape = tuple(base["image"]["standardized_shape"][:2])
    rgb_centroid = (float(base["visual_field"]["delta_x"]), float(base["visual_field"]["delta_y"]))
    rgb_in_region = base["region"]["target_mass_ratio"] >= (profile.min_target_mass_ratio or 0.5)
    reads = [Read("rgb_structure", "rgb", rgb_centroid, bool(rgb_in_region),
                  note=f"isolate={base['isolation']['strategy']}")]

    if depth_path is not None:
        depth = load_depth_map(depth_path, target_shape=shape)
        reads.append(_depth_occupancy_read(depth, table_z, profile.target_region))

    verdict = fuse_reads(reads)
    return {
        "schema_version": "vtl-parallax-success/v1",
        "task": base["task"],
        "parallax": {
            "success": verdict.success,
            "confidence": verdict.confidence,
            "omega": verdict.omega,
            "region_agreement": verdict.region_agreement,
            "unanimous": verdict.unanimous,
            "independence": verdict.independence,
            "independence_note": verdict.independence_note,
            "n_reads": verdict.n_reads,
            "reads": verdict.detail["per_read"],
        },
        "rgb_report": base,
    }


def report_to_json(report: dict[str, Any]) -> str:
    def default(obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    return json.dumps(report, indent=2, sort_keys=True, default=default)

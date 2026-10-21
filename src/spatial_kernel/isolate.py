"""Deterministic object isolation.

The original kernel measures the centroid of *all* gradient energy in the frame.
In a clean single-object scene that is the object. In any cluttered real scene it
is table edges, shadows, background texture and the robot arm — so `delta_x/delta_y`
report "where is the scene's edge energy", not "where is the manipuland". A verifier
built on that confound fails silently and confidently.

This module isolates the task-relevant object *before* the kernel runs, without
introducing a learned judge. Perception may be fed a mask; the verdict downstream
stays deterministic. Three strategies, in declared order of trust:

  1. ``explicit``        — a caller-supplied binary object mask (e.g. from SAM2,
                           a depth sensor's segmentation, or a sim renderer). The
                           segmenter is allowed to be learned; the kernel that
                           judges is not.
  2. ``depth_plane``     — remove the support plane (``table_z``) and keep what
                           stands above it. Pure geometry, no learning.
  3. ``largest_component`` — keep the largest connected gradient blob. A weak,
                           assumption-heavy fallback for single-object scenes only;
                           it is reported as low-confidence.

Every isolation returns provenance: which strategy fired, how much mass it kept,
and a confidence tier. Honesty about *how* the object was found travels with the
number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from skimage.measure import label


@dataclass(frozen=True)
class IsolationResult:
    object_mask: np.ndarray
    strategy: str
    confidence: str  # "high" | "medium" | "low" | "none"
    kept_mass_fraction: float  # object mass / original gradient mass
    notes: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)


def _coerce_binary(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[:, :, 0]
    if m.shape != shape:
        # nearest-neighbour resize without pulling in cv2 here; mask is binary
        ys = (np.linspace(0, m.shape[0] - 1, shape[0])).round().astype(int)
        xs = (np.linspace(0, m.shape[1] - 1, shape[1])).round().astype(int)
        m = m[np.ix_(ys, xs)]
    return (m > 0).astype(np.uint8)


def isolate_from_explicit_mask(
    gradient_mask: np.ndarray,
    object_mask: np.ndarray,
) -> IsolationResult:
    obj = _coerce_binary(object_mask, gradient_mask.shape)
    grad = (gradient_mask > 0).astype(np.uint8)
    # The kernel reads structure, so keep gradient energy that falls on the object.
    kept = (grad & obj).astype(np.uint8)
    # If the object mask covers a flat (gradient-free) object, fall back to the
    # object silhouette itself so the kernel still has mass to measure.
    if int(kept.sum()) == 0 and int(obj.sum()) > 0:
        kept = obj
        notes = "object surface carried no gradient energy; using silhouette."
    else:
        notes = ""
    grad_total = float(grad.sum())
    kept_frac = float(kept.sum() / grad_total) if grad_total > 0 else 0.0
    return IsolationResult(
        object_mask=kept,
        strategy="explicit",
        confidence="high",
        kept_mass_fraction=kept_frac,
        notes=notes,
        provenance={"object_mask_pixels": int(obj.sum())},
    )


def isolate_from_depth(
    gradient_mask: np.ndarray,
    depth: np.ndarray,
    table_z: float | None = None,
    min_height: float = 1e-3,
) -> IsolationResult:
    grad = (gradient_mask > 0).astype(np.uint8)
    valid = np.isfinite(depth)
    if not np.any(valid):
        return IsolationResult(
            object_mask=grad,
            strategy="depth_plane",
            confidence="none",
            kept_mass_fraction=1.0,
            notes="no finite depth samples; isolation skipped.",
        )
    if table_z is None:
        # 5th percentile = the support surface (most pixels are table-or-below).
        table_z = float(np.nanpercentile(depth[valid], 5.0))
    above = valid & ((depth - table_z) > min_height)
    obj = (above.astype(np.uint8))
    kept = (grad & obj).astype(np.uint8)
    if int(kept.sum()) == 0 and int(obj.sum()) > 0:
        kept = obj
        notes = "above-plane object carried no gradient energy; using occupancy."
    else:
        notes = ""
    grad_total = float(grad.sum())
    kept_frac = float(kept.sum() / grad_total) if grad_total > 0 else 0.0
    return IsolationResult(
        object_mask=kept,
        strategy="depth_plane",
        confidence="high",
        kept_mass_fraction=kept_frac,
        notes=notes,
        provenance={"table_z": float(table_z), "above_plane_pixels": int(obj.sum())},
    )


def isolate_largest_component(gradient_mask: np.ndarray) -> IsolationResult:
    grad = (gradient_mask > 0).astype(np.uint8)
    lab = label(grad, connectivity=2)
    n = int(lab.max())
    if n == 0:
        return IsolationResult(
            object_mask=grad,
            strategy="largest_component",
            confidence="none",
            kept_mass_fraction=1.0,
            notes="no gradient mass to isolate.",
        )
    counts = np.bincount(lab.ravel())
    counts[0] = 0  # background
    biggest = int(counts.argmax())
    kept = (lab == biggest).astype(np.uint8)
    grad_total = float(grad.sum())
    kept_frac = float(kept.sum() / grad_total) if grad_total > 0 else 0.0
    # If the "largest" blob is a thin sliver of the scene, this assumption is unsafe.
    confidence = "medium" if kept_frac >= 0.20 else "low"
    return IsolationResult(
        object_mask=kept,
        strategy="largest_component",
        confidence=confidence,
        kept_mass_fraction=kept_frac,
        notes="single-object assumption; unsafe in clutter.",
        provenance={"n_components": n, "kept_component": biggest},
    )


def isolate_object(
    gradient_mask: np.ndarray,
    *,
    object_mask: np.ndarray | None = None,
    depth: np.ndarray | None = None,
    table_z: float | None = None,
    strategy: str = "auto",
    min_height: float = 1e-3,
) -> IsolationResult:
    """Isolate the manipuland from whole-frame gradient mass.

    ``strategy="auto"`` picks the most trustworthy available source:
    explicit mask > depth plane > largest component. Pass an explicit strategy to
    force one.
    """
    if strategy not in {"auto", "explicit", "depth_plane", "largest_component", "none"}:
        raise ValueError(f"Unknown isolation strategy: {strategy}")

    if strategy == "none":
        grad = (gradient_mask > 0).astype(np.uint8)
        return IsolationResult(
            object_mask=grad,
            strategy="none",
            confidence="none",
            kept_mass_fraction=1.0,
            notes="isolation disabled; whole-frame gradient mass (legacy behaviour).",
        )

    if strategy in {"auto", "explicit"} and object_mask is not None:
        return isolate_from_explicit_mask(gradient_mask, object_mask)
    if strategy == "explicit":
        raise ValueError("strategy='explicit' requires object_mask")

    if strategy in {"auto", "depth_plane"} and depth is not None:
        return isolate_from_depth(gradient_mask, depth, table_z=table_z, min_height=min_height)
    if strategy == "depth_plane":
        raise ValueError("strategy='depth_plane' requires a depth map")

    return isolate_largest_component(gradient_mask)

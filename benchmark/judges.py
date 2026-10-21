"""Three judges over the same outcomes. Each returns (verdict_pass, confidence, note).

appearance_only : deterministic proxy for a geometry-blind judge. Given a perfect
                  segmenter mask, it asks only "is the object in the target region
                  in the picture?" It cannot use depth — so it cannot tell a real
                  object from a flat printed decoy, and cannot confirm an occluded
                  object is actually present. This is the honest stand-in for the
                  failure mode any RGB-only judge (including a VLM looking at one
                  frame) shares: appearance does not carry geometry.

parallax        : ours. Independent RGB + depth reads must agree, with Ω as
                  confidence. (spatial_kernel.verify_parallax)

vlm             : OPTIONAL real Gemini Robotics-ER call via spatial_kernel.robotics_er.
                  Runs only if a key is supplied; otherwise reported as skipped.
                  No numbers are fabricated for it.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from spatial_kernel.core import compute_kernel_metrics_from_bgr
from spatial_kernel.regions import region_mask, region_metrics
from spatial_kernel.tasks import get_task_profile
from spatial_kernel.verify import verify_parallax

from .scenes import TARGET_REGION, Scene


def judge_appearance_only(scene: Scene) -> tuple[bool, float, str]:
    """RGB-only: object (segmented) centroid in target region, with mass support."""
    profile = get_task_profile(TARGET_REGION)
    if scene.object_mask is None or int(scene.object_mask.sum()) == 0:
        return False, 0.9, "no object segmented"
    metrics, fields = compute_kernel_metrics_from_bgr(
        np.ascontiguousarray(scene.rgb), object_mask=scene.object_mask
    )
    region = region_metrics(fields["mass_mask"], profile.target_region)
    rmask = region_mask(fields["mass_mask"].shape, profile.target_region)
    ys, xs = np.nonzero(fields["mass_mask"])
    if len(xs) == 0:
        return False, 0.5, "empty mass"
    in_region = bool(rmask[int(ys.mean()), int(xs.mean())])
    support = region["target_mass_ratio"] >= (profile.min_target_mass_ratio or 0.5)
    verdict = in_region and support
    # An appearance judge is *confident*: the picture looks correct. That false
    # confidence on the decoy is exactly the point.
    return verdict, 0.85 if verdict else 0.7, f"in_region={in_region} support={support}"


def judge_parallax(scene: Scene) -> tuple[bool, float, str]:
    with tempfile.TemporaryDirectory() as d:
        import cv2
        ip, dp = Path(d) / "s.png", Path(d) / "d.npy"
        cv2.imwrite(str(ip), scene.rgb)
        np.save(str(dp), scene.depth)
        out = verify_parallax(
            str(ip), task=TARGET_REGION, depth_path=str(dp), table_z=0.0,
            rgb_isolate="explicit" if scene.object_mask is not None else "largest_component",
            object_mask=scene.object_mask,
        )
    p = out["parallax"]
    return bool(p["success"]), float(p["confidence"]), f"Ω={p['omega']:.3f} agree={p['region_agreement']:.2f}"


def judge_vlm(scene: Scene, api_key: str | None) -> tuple[bool, float, str] | None:
    """Real Gemini Robotics-ER judge — RGB-only, its own verdict. None when skipped.

    Fair setup: the VLM sees the same single RGB frame the appearance judge sees;
    only parallax additionally sees depth. An unparseable/refused answer is counted
    conservatively as a FAIL with a note (it could not certify the outcome).
    """
    if not api_key:
        return None
    from .vlm_judge import vlm_judge_rgb

    verdict, text = vlm_judge_rgb(scene.rgb, api_key)
    one_line = " ".join(text.split())[:90]
    if verdict is None:
        return False, 0.3, f"unparseable/refused: {one_line}"
    return verdict, 0.75, one_line

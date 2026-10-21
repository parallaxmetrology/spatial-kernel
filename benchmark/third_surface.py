"""Third independent surface: proprioception (grasp), and the 3-surface verdict.

The v2 benchmark showed 2 vision surfaces (RGB + depth) have an adversarial complement:
they false-FAIL real objects the depth camera can't see (transparent, occluded-from-cam)
and false-PASS a 3D decoy both cameras agree on (raised_decoy). The Parallax thesis says
robustness scales with the number of GENUINELY INDEPENDENT surfaces. The canonical third
surface in robotics is proprioception — the gripper's own sense of whether it closed on a
real object of the expected profile. It is touch, not vision, so it is maximally
independent of both cameras, and it is exactly the signal the camera failures lack.

Honesty note: proprioception here is MODELED from each scene's true physical grasp height
(`Scene.true_grasp_height`), which is independent of the depth-camera reading. We are not
claiming a real gripper — we are demonstrating that a third surface carrying the physical
fact the cameras miss closes the holes. A real gripper supplies this signal directly.

Decision rule (principled, not vote-counting):
    localized   = RGB places it at target OR depth places it at target
    real_object = proprioception confirms graspable mass of the EXPECTED profile at target
    success     = localized AND real_object

- catches flat decoy (no graspable mass) and raised_decoy (wrong profile)
- rescues transparent / dropout (depth camera blind, but RGB localizes and the gripper
  confirms a real object) — a surface that is out of its operating envelope does not veto.
"""

from __future__ import annotations

import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from spatial_kernel.regions import region_center_normalized
from spatial_kernel.verify import verify_parallax

from .scenes import CLASSES, EXPECTED_OBJECT_HEIGHT, TARGET_REGION, Scene, generate

MIN_GRASPABLE = 0.03   # below this, the gripper finds nothing
HEIGHT_TOL = 0.06      # tolerance on matching the expected object profile


def proprioception_in_region(scene: Scene) -> tuple[bool, str]:
    h = scene.true_grasp_height
    if h < MIN_GRASPABLE:
        return False, f"grasp_h={h:.2f}: nothing graspable"
    if abs(h - EXPECTED_OBJECT_HEIGHT) > HEIGHT_TOL:
        return False, f"grasp_h={h:.2f}: wrong profile (expected {EXPECTED_OBJECT_HEIGHT:.2f})"
    return True, f"grasp_h={h:.2f}: real object, profile match"


def verdicts(scene: Scene) -> dict:
    """Return 2-surface and 3-surface verdicts for one scene."""
    with tempfile.TemporaryDirectory() as d:
        ip, dp = Path(d) / "s.png", Path(d) / "d.npy"
        cv2.imwrite(str(ip), scene.rgb)
        np.save(str(dp), scene.depth)
        out = verify_parallax(
            str(ip), task=TARGET_REGION, depth_path=str(dp), table_z=0.0,
            rgb_isolate="explicit" if scene.object_mask is not None else "largest_component",
            object_mask=scene.object_mask,
        )
    reads = {r["surface"]: r for r in out["parallax"]["reads"]}
    rgb_in = bool(reads.get("rgb", {}).get("in_region", False))
    depth_in = bool(reads.get("depth", {}).get("in_region", False))
    two_surface = bool(out["parallax"]["success"])

    prop_in, prop_note = proprioception_in_region(scene)
    localized = rgb_in or depth_in
    three_surface = bool(localized and prop_in)

    # coherence view: with the third surface, is the verdict unanimous?
    votes = [rgb_in, depth_in, prop_in]
    agreement = sum(votes) / 3.0
    return {
        "cls": scene.cls, "gt": scene.gt_pass,
        "rgb_in": rgb_in, "depth_in": depth_in, "prop_in": prop_in, "prop_note": prop_note,
        "two_surface": two_surface, "three_surface": three_surface,
        "agreement_3": agreement,
    }


def main() -> None:
    per_class = 40
    scenes = generate(per_class=per_class, seed=21)
    rows = [verdicts(sc) for sc in scenes]
    n = len(rows)

    acc2 = sum((r["two_surface"] == r["gt"]) for r in rows) / n
    acc3 = sum((r["three_surface"] == r["gt"]) for r in rows) / n
    fp2 = sum(1 for r in rows if r["two_surface"] and not r["gt"])
    fp3 = sum(1 for r in rows if r["three_surface"] and not r["gt"])

    print("=" * 70)
    print(f"THIRD SURFACE  ·  proprioception added  ·  {n} scenes ({per_class}/class)")
    print("=" * 70)
    print(f"\nOVERALL ACCURACY   2-surface (RGB+depth): {acc2:.1%}    3-surface (+grasp): {acc3:.1%}")
    print(f"FALSE PASSES       2-surface: {fp2}    3-surface: {fp3}")

    by = defaultdict(list)
    for r in rows:
        by[r["cls"]].append(r)
    print(f"\n  {'class':<22}{'GT':>5}{'2-surf':>9}{'3-surf':>9}   note")
    for c in CLASSES:
        rs = by[c]
        a2 = sum(x["two_surface"] == x["gt"] for x in rs) / len(rs)
        a3 = sum(x["three_surface"] == x["gt"] for x in rs) / len(rs)
        gt = "PASS" if rs[0]["gt"] else "FAIL"
        flag = "  <- fixed by grasp" if a3 > a2 + 1e-9 else ""
        print(f"  {c:<22}{gt:>5}{a2:>8.0%}{a3:>8.0%}   {rs[0]['prop_note']}{flag}")

    print("\n  Mechanism on the three holes:")
    for c in ("transparent_success", "depth_dropout_success", "raised_decoy"):
        r = by[c][0]
        print(f"   {c:<22} rgb_in={r['rgb_in']} depth_in={r['depth_in']} grasp_in={r['prop_in']} "
              f"-> 3-surface={'PASS' if r['three_surface'] else 'FAIL'} (gt {'PASS' if r['gt'] else 'FAIL'})")
    print("=" * 70)


if __name__ == "__main__":
    main()

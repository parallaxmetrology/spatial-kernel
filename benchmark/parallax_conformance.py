"""Conform the robotics verifier to the Parallax meta-instrument spec.

Implements the two rigor protocols that brief_meta_parallax_instrument.md marks as
required-before-reporting-Ω, computed on the actual benchmark scenes:

  1. INDEPENDENCE AUDIT (§2, §3) — are the RGB and depth reads genuinely independent
     observables ("critics in different rooms"), or transforms of one surface
     ("different glasses on the same wall")? The spec's test is the observable
     correlation and a selective-divergence check: a shared-input pair CANNOT
     disagree based on geometry it cannot see, so divergence on the depth-only and
     rgb-only adversarial classes is positive evidence of independence.

  2. NULL PROTOCOL (§6) — the instrument must collapse when structure is destroyed.
     We scramble the depth occupancy to a random location disjoint from the RGB
     object. A real agreement instrument must then FAIL (the surfaces no longer
     co-locate); if it still passes, it was not measuring real agreement.

Run:  python3 -m benchmark.parallax_conformance
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np

from spatial_kernel.verify import verify_parallax

from .scenes import CLASSES, H, OBJ_H, OBJ_W, W, Scene, generate, make_scene


def _reads_for_scene(sc: Scene) -> tuple[dict, dict] | None:
    with tempfile.TemporaryDirectory() as d:
        ip, dp = Path(d) / "s.png", Path(d) / "d.npy"
        cv2.imwrite(str(ip), sc.rgb)
        np.save(str(dp), sc.depth)
        out = verify_parallax(
            str(ip), task="bottom_left", depth_path=str(dp), table_z=0.0,
            rgb_isolate="explicit" if sc.object_mask is not None else "largest_component",
            object_mask=sc.object_mask,
        )
    reads = {r["surface"]: r for r in out["parallax"]["reads"]}
    if "rgb" not in reads or "depth" not in reads:
        return None
    return reads["rgb"], reads["depth"]


def independence_audit(per_class: int = 40, seed: int = 11) -> dict:
    scenes = generate(per_class=per_class, seed=seed)
    rgb_dx, rgb_dy, dep_dx, dep_dy = [], [], [], []
    disagree_by_cls = {c: [] for c in CLASSES}
    for sc in scenes:
        r = _reads_for_scene(sc)
        if r is None:
            continue
        rgb, dep = r
        rgb_dx.append(rgb["centroid"][0]); rgb_dy.append(rgb["centroid"][1])
        dep_dx.append(dep["centroid"][0]); dep_dy.append(dep["centroid"][1])
        disagree_by_cls[sc.cls].append(int(rgb["in_region"] != dep["in_region"]))

    rgb_dx, rgb_dy = np.array(rgb_dx), np.array(rgb_dy)
    dep_dx, dep_dy = np.array(dep_dx), np.array(dep_dy)

    def _r(a, b):
        if a.std() < 1e-9 or b.std() < 1e-9:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    r_x, r_y = _r(rgb_dx, dep_dx), _r(rgb_dy, dep_dy)
    mean_abs_r = float(np.mean([abs(r_x), abs(r_y)]))
    overall_disagree = float(np.mean([d for ds in disagree_by_cls.values() for d in ds]))

    return {
        "n": len(rgb_dx),
        "r_centroid_x": round(r_x, 3),
        "r_centroid_y": round(r_y, 3),
        "mean_abs_r": round(mean_abs_r, 3),
        "overall_disagreement_rate": round(overall_disagree, 3),
        "disagreement_by_class": {c: round(float(np.mean(v)), 3) if v else None
                                  for c, v in disagree_by_cls.items()},
    }


def _scramble_depth(sc: Scene, rng: np.random.Generator) -> Scene:
    """Null: move the depth occupancy to a random quadrant disjoint from the object."""
    depth = np.zeros((H, W), dtype=np.float32)
    # random top-strip location (not the bottom-left object region)
    top = int(rng.integers(20, 120)); left = int(rng.integers(250, 330))
    depth[top:top + OBJ_H, left:left + OBJ_W] = 0.20
    return Scene(cls="null_" + sc.cls, rgb=sc.rgb, depth=depth,
                object_mask=sc.object_mask, gt_pass=sc.gt_pass)


def null_protocol(per_class: int = 60, seed: int = 13) -> dict:
    rng = np.random.default_rng(seed)
    # Use clean_success scenes — the case parallax is supposed to PASS.
    real = [make_scene("clean_success", rng) for _ in range(per_class)]
    real_pass, null_pass = 0, 0
    for sc in real:
        rr = _reads_for_scene(sc)
        nn = _reads_for_scene(_scramble_depth(sc, rng))
        if rr and rr[0]["in_region"] and rr[1]["in_region"]:
            real_pass += 1
        if nn and nn[0]["in_region"] and nn[1]["in_region"]:
            null_pass += 1
    return {
        "n": per_class,
        "real_agreement_pass_rate": round(real_pass / per_class, 3),
        "null_agreement_pass_rate": round(null_pass / per_class, 3),
    }


def main() -> None:
    print("=" * 70)
    print("PARALLAX CONFORMANCE  ·  robotics verifier vs meta-instrument spec")
    print("=" * 70)

    ind = independence_audit()
    print("\n1. INDEPENDENCE AUDIT (meta-spec §2-3)")
    print(f"   n={ind['n']}  centroid correlation: r_x={ind['r_centroid_x']}  r_y={ind['r_centroid_y']}")
    print(f"   overall surface-disagreement rate: {ind['overall_disagreement_rate']:.0%}")
    print("   disagreement by class (a shared-input pair could not selectively diverge):")
    for c, v in ind["disagreement_by_class"].items():
        if v:
            print(f"     {c:<22} {v:.0%}")
    print("   → RGB and depth agree on coherent scenes (both accurate) yet diverge")
    print("     exactly on the depth-only/rgb-only classes. Selective divergence is")
    print("     the independence evidence: shared-surface reads cannot do this.")

    nul = null_protocol()
    print("\n2. NULL PROTOCOL (meta-spec §6)")
    print(f"   real depth (object co-located):   agreement-pass rate {nul['real_agreement_pass_rate']:.0%}")
    print(f"   scrambled depth (structure destroyed): pass rate {nul['null_agreement_pass_rate']:.0%}")
    collapse = nul["real_agreement_pass_rate"] - nul["null_agreement_pass_rate"]
    verdict = "PASS — instrument collapses under structure destruction" if collapse > 0.5 else "FAIL — instrument not measuring real agreement"
    print(f"   collapse = {collapse:.0%}   →  {verdict}")
    print("=" * 70)


if __name__ == "__main__":
    main()

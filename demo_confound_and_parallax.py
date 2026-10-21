"""Synthetic proof: the gradient-mass confound, the isolation fix, and parallax Ω.

Controlled scene with known ground truth (the repo's null-protocol style):

  - a single solid manipuland in the bottom-left  (the object the robot placed)
  - many small distractor specks in the top-right  (background clutter / texture)
  - a depth map where ONLY the manipuland stands above the table plane

We then show three things with numbers:

  1. CONFOUND  — whole-frame gradient centroid is pulled toward the clutter, away
                 from the true object. The legacy verifier measures scene texture.
  2. FIX       — depth-plane (and largest-component) isolation recover the true
                 object centroid to within a few percent of ground truth.
  3. PARALLAX  — RGB and depth are independent surfaces; when they agree, Ω is low
                 and confidence high; when an attacker satisfies one surface but
                 not the other, Ω spikes and confidence collapses — the outcome is
                 correctly reported as contested rather than passed.

Run:  python3 demo_confound_and_parallax.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np

from spatial_kernel.verify import verify_parallax, verify_spatial_task


H = W = 512
OBJ_ROWS = (330, 470)
OBJ_COLS = (40, 200)


def ground_truth_centroid() -> tuple[float, float]:
    cx = (OBJ_COLS[0] + OBJ_COLS[1]) / 2.0
    cy = (OBJ_ROWS[0] + OBJ_ROWS[1]) / 2.0
    return ((cx - (W - 1) / 2.0) / W, (cy - (H - 1) / 2.0) / H)


def build_scene(seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), 245, dtype=np.uint8)

    # Manipuland: one solid dark block in the bottom-left (largest connected blob).
    cv2.rectangle(img, (OBJ_COLS[0], OBJ_ROWS[0]), (OBJ_COLS[1], OBJ_ROWS[1]), (25, 25, 25), -1)
    cv2.line(img, (OBJ_COLS[0] + 10, OBJ_ROWS[0] + 15),
             (OBJ_COLS[1] - 10, OBJ_ROWS[1] - 15), (210, 210, 210), 3)

    # Distractor clutter: many small disconnected specks in the top-right.
    for _ in range(220):
        x = int(rng.integers(285, 500))
        y = int(rng.integers(20, 235))
        s = int(rng.integers(2, 6))
        shade = int(rng.integers(30, 90))
        cv2.rectangle(img, (x, y), (x + s, y + s), (shade, shade, shade), -1)

    # Depth: table at 0, ONLY the manipuland stands above the plane. The painted
    # clutter is flat texture (height 0) — invisible to a geometric read.
    depth = np.zeros((H, W), dtype=np.float32)
    depth[OBJ_ROWS[0]:OBJ_ROWS[1], OBJ_COLS[0]:OBJ_COLS[1]] = 0.20
    return img, depth


def _fmt(c: tuple[float, float]) -> str:
    return f"(dx={c[0]:+.3f}, dy={c[1]:+.3f})"


def main() -> None:
    gx, gy = ground_truth_centroid()
    print("=" * 72)
    print("GROUND TRUTH manipuland centroid:", _fmt((gx, gy)), "  (bottom-left)")
    print("=" * 72)

    with tempfile.TemporaryDirectory() as d:
        img_path = Path(d) / "scene.png"
        depth_path = Path(d) / "depth.npy"
        img, depth = build_scene()
        cv2.imwrite(str(img_path), img)
        np.save(str(depth_path), depth)

        # 1. CONFOUND — legacy whole-frame read.
        legacy = verify_spatial_task(str(img_path), task="bottom_left", isolate="none")
        lc = (legacy["visual_field"]["delta_x"], legacy["visual_field"]["delta_y"])
        lerr = float(np.hypot(lc[0] - gx, lc[1] - gy))
        print("\n1. CONFOUND  (isolate=none, whole-frame gradient mass)")
        print(f"   measured centroid {_fmt(lc)}   error={lerr:.3f}   status={legacy['compliance']['status']}")
        print("   → pulled toward top-right clutter; this is scene texture, not the object.")

        # 2. FIX — depth-plane and largest-component isolation.
        dep = verify_spatial_task(str(img_path), task="bottom_left",
                                  depth_path=str(depth_path), table_z=0.0, isolate="depth_plane")
        dc = (dep["visual_field"]["delta_x"], dep["visual_field"]["delta_y"])
        derr = float(np.hypot(dc[0] - gx, dc[1] - gy))
        lcomp = verify_spatial_task(str(img_path), task="bottom_left", isolate="largest_component")
        kc = (lcomp["visual_field"]["delta_x"], lcomp["visual_field"]["delta_y"])
        kerr = float(np.hypot(kc[0] - gx, kc[1] - gy))
        print("\n2. FIX  (object isolated before the kernel runs)")
        print(f"   depth_plane        {_fmt(dc)}   error={derr:.3f}   status={dep['compliance']['status']}")
        print(f"   largest_component  {_fmt(kc)}   error={kerr:.3f}   status={lcomp['compliance']['status']}")
        print(f"   → isolation cuts centroid error {lerr:.3f} → {derr:.3f} "
              f"({(1 - derr / max(lerr, 1e-9)) * 100:.0f}% closer to ground truth).")

        # 3. PARALLAX — agreement of independent surfaces.
        # Production architecture: learned eyes (a segmenter mask for the RGB
        # surface) + deterministic judge. Depth stays a separate geometric surface.
        obj_silhouette = np.zeros((H, W), dtype=np.uint8)
        obj_silhouette[OBJ_ROWS[0]:OBJ_ROWS[1], OBJ_COLS[0]:OBJ_COLS[1]] = 1
        good = verify_parallax(str(img_path), task="bottom_left",
                               depth_path=str(depth_path), table_z=0.0,
                               rgb_isolate="explicit", object_mask=obj_silhouette)
        p = good["parallax"]
        print("\n3. PARALLAX  (RGB structure + depth occupancy, independent surfaces)")
        print(f"   reads: {[r['name'] + ':' + ('in' if r['in_region'] else 'out') for r in p['reads']]}")
        print(f"   Ω={p['omega']:.3f}  region_agreement={p['region_agreement']:.2f}  "
              f"independence={p['independence']}  confidence={p['confidence']:.2f}  success={p['success']}")
        print(f"   {p['independence_note']}")

        # Attacker: an outcome that satisfies the RGB read but NOT depth (e.g. a
        # flat printed photo of the object in the corner — looks right, no geometry).
        from spatial_kernel.parallax import Read, fuse_reads
        rgb_only = Read("rgb_structure", "rgb", (gx, gy), True, note="printed decoy")
        depth_says_empty = Read("depth_occupancy", "depth", (0.30, -0.30), False, note="nothing above plane there")
        attacked = fuse_reads([rgb_only, depth_says_empty])
        print("\n   ATTACK  (RGB satisfied by a flat decoy; depth sees no object there)")
        print(f"   Ω={attacked.omega:.3f}  region_agreement={attacked.region_agreement:.2f}  "
              f"confidence={attacked.confidence:.2f}  success={attacked.success}")
        print("   → disagreement spikes; the verdict is contested, not passed. "
              "A single learned judge could be fooled here.")
        print("=" * 72)


if __name__ == "__main__":
    main()

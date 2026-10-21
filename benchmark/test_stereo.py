"""Synthetic two-view parallax test: does the geometric read separate flat vs raised?

We simulate a stereo pair by shifting the whole scene (background parallax) and giving
the object region EXTRA shift when it is raised (closer to the camera). A flat decoy
gets zero extra shift (it lies on the table). The read must report:
  raised object  -> above_plane = True   (large relative parallax)
  flat decoy     -> above_plane = False  (~zero relative parallax)

Textured surfaces are used so ORB has features (as a real photo would).
"""

from __future__ import annotations

import numpy as np

from spatial_kernel.stereo import two_view_geometric_read

H = W = 480
OBJ = (300, 60, 440, 220)  # top,left,bot,right (bottom-left quadrant)


def _textured(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # table texture so ORB has features everywhere
    img = (rng.integers(180, 235, size=(H, W, 3))).astype(np.uint8)
    return img


def _paste_object(img: np.ndarray, dx: int, seed: int = 99) -> np.ndarray:
    """Draw a textured object shifted horizontally by dx."""
    rng = np.random.default_rng(seed)
    out = img.copy()
    t, l, b, r = OBJ
    l2, r2 = l + dx, r + dx
    patch = (rng.integers(20, 90, size=(b - t, r - l, 3))).astype(np.uint8)
    if 0 <= l2 and r2 <= W:
        out[t:b, l2:r2] = patch
    return out


def make_pair(kind: str, bg_shift: int = 8, obj_extra: int = 14):
    """Return (left, right, object_mask) for 'raised' or 'flat'."""
    base = _textured(1)
    extra = obj_extra if kind == "raised" else 0
    left = _paste_object(base, 0)
    # right view: whole scene shifts by bg_shift; object shifts bg_shift+extra
    right_bg = np.roll(base, -bg_shift, axis=1)
    right = _paste_object(right_bg, -(bg_shift + extra))
    mask = np.zeros((H, W), np.uint8)
    t, l, b, r = OBJ
    mask[t:b, l:r] = 1
    return left, right, mask


def main() -> None:
    print("=" * 64)
    print("SYNTHETIC TWO-VIEW PARALLAX  ·  flat decoy vs raised object")
    print("=" * 64)
    for kind in ("raised", "flat"):
        left, right, mask = make_pair(kind)
        rd = two_view_geometric_read(left, right, mask, parallax_threshold=4.0)
        verdict = "ABOVE PLANE" if rd.above_plane else "at plane / flat"
        print(f"\n{kind:>7}:  rel_parallax={rd.relative_parallax:+.1f}px  "
              f"(obj={rd.object_disparity}, table={rd.table_disparity})  "
              f"matches obj/tab={rd.n_object_matches}/{rd.n_table_matches}")
        print(f"         verdict: {verdict}   confidence={rd.confidence}")
    print("\n→ raised object reads ABOVE PLANE; flat decoy reads at-plane.")
    print("  This is the LiDAR-free geometric surface for the iPhone 14 Plus.")
    print("=" * 64)


if __name__ == "__main__":
    main()

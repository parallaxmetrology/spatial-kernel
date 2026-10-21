"""Two-view parallax: the independent geometric surface, without LiDAR.

On a phone with no depth sensor (e.g. iPhone 14 Plus), the geometric observable the
parallax verifier needs is recovered the way the instrument is named for: from the
apparent displacement of the scene between two viewpoints.

We do NOT need a dense metric depth map. The discriminator we need is binary and
robust: does the object region show *more* parallax than the surrounding table?
A raised real object is closer to the camera, so its features shift more between the
two shots than the table plane does. A flat printed decoy lying on the table shifts
*with* the table — zero relative parallax. A printed photo on a box (raised_decoy)
shifts like a raised object — which is exactly why that case still needs a third
surface, and the read honestly reports it as "raised", not "real".

Method (no calibration, no rectification — robust to handheld capture):
  1. ORB features + matching between left/right.
  2. Horizontal disparity per match (x_left - x_right).
  3. Partition matches into object (inside mask) vs table (outside).
  4. relative_parallax = median|object disparity| - median|table disparity|.
  5. above_plane if relative_parallax exceeds a calibrated threshold.

The threshold is scene-scale dependent (baseline, object height, distance) and must
be calibrated once with a reference shot — exactly the operating-envelope discipline
the Parallax meta-spec requires (state scale before reporting).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class StereoRead:
    relative_parallax: float      # object median |disparity| - table median |disparity|, px
    object_disparity: float
    table_disparity: float
    n_object_matches: int
    n_table_matches: int
    above_plane: bool
    confidence: float
    note: str = ""


def two_view_geometric_read(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    object_mask: np.ndarray,
    parallax_threshold: float = 4.0,
    max_features: int = 4000,
) -> StereoRead:
    gray_l = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=max_features)
    kp_l, des_l = orb.detectAndCompute(gray_l, None)
    kp_r, des_r = orb.detectAndCompute(gray_r, None)
    if des_l is None or des_r is None or len(kp_l) < 8 or len(kp_r) < 8:
        return StereoRead(0, 0, 0, 0, 0, False, 0.0, "insufficient features")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_l, des_r)
    obj = np.asarray(object_mask) > 0
    h, w = obj.shape

    obj_disp, tab_disp = [], []
    for m in matches:
        xl, yl = kp_l[m.queryIdx].pt
        xr, yr = kp_r[m.trainIdx].pt
        # reject vertical outliers (rough epipolar sanity for near-parallel capture)
        if abs(yl - yr) > 0.04 * h:
            continue
        dx = xl - xr
        iy, ix = int(round(yl)), int(round(xl))
        if 0 <= iy < h and 0 <= ix < w and obj[iy, ix]:
            obj_disp.append(dx)
        else:
            tab_disp.append(dx)

    if len(obj_disp) < 4 or len(tab_disp) < 4:
        return StereoRead(0, 0, len(obj_disp), len(tab_disp), 0, False, 0.0,
                          "too few partitioned matches")

    od = float(np.median(np.abs(obj_disp)))
    td = float(np.median(np.abs(tab_disp)))
    rel = od - td
    above = rel > parallax_threshold
    # confidence grows with match support and margin past threshold
    support = min(1.0, (len(obj_disp) + len(tab_disp)) / 200.0)
    margin = min(1.0, abs(rel) / (2.0 * parallax_threshold))
    conf = round(0.5 * support + 0.5 * margin, 3)
    return StereoRead(round(rel, 2), round(od, 2), round(td, 2),
                      len(obj_disp), len(tab_disp), above, conf,
                      f"thresh={parallax_threshold}")


def stereo_read_to_parallax_read(stereo: StereoRead, region_centroid_norm, in_region: bool):
    """Adapt a StereoRead into a parallax.Read on the 'depth' surface.

    The geometric surface contributes its in_region verdict only if it sees an object
    above the plane; otherwise it reports out-of-region (no object present there).
    """
    from .parallax import Read
    geom_in = bool(in_region and stereo.above_plane)
    return Read(
        name="two_view_parallax", surface="depth",
        centroid=region_centroid_norm, in_region=geom_in,
        weight=1.0, note=f"rel_parallax={stereo.relative_parallax}px above_plane={stereo.above_plane}",
    )

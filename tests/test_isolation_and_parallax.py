import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from spatial_kernel.parallax import Read, fuse_reads
from spatial_kernel.verify import verify_parallax, verify_spatial_task

H = W = 512
OBJ_ROWS = (330, 470)
OBJ_COLS = (40, 200)


def _gt_centroid():
    cx = (OBJ_COLS[0] + OBJ_COLS[1]) / 2.0
    cy = (OBJ_ROWS[0] + OBJ_ROWS[1]) / 2.0
    return ((cx - (W - 1) / 2.0) / W, (cy - (H - 1) / 2.0) / H)


def _build(d: Path):
    rng = np.random.default_rng(7)
    img = np.full((H, W, 3), 245, dtype=np.uint8)
    cv2.rectangle(img, (OBJ_COLS[0], OBJ_ROWS[0]), (OBJ_COLS[1], OBJ_ROWS[1]), (25, 25, 25), -1)
    for _ in range(220):
        x = int(rng.integers(285, 500)); y = int(rng.integers(20, 235))
        s = int(rng.integers(2, 6)); shade = int(rng.integers(30, 90))
        cv2.rectangle(img, (x, y), (x + s, y + s), (shade, shade, shade), -1)
    depth = np.zeros((H, W), dtype=np.float32)
    depth[OBJ_ROWS[0]:OBJ_ROWS[1], OBJ_COLS[0]:OBJ_COLS[1]] = 0.20
    ip, dp = d / "scene.png", d / "depth.npy"
    cv2.imwrite(str(ip), img)
    np.save(str(dp), depth)
    return ip, dp


class IsolationTests(unittest.TestCase):
    def test_wholeframe_is_confounded_by_clutter(self):
        with tempfile.TemporaryDirectory() as d:
            ip, _ = _build(Path(d))
            r = verify_spatial_task(str(ip), task="bottom_left", isolate="none")
            gx, gy = _gt_centroid()
            c = (r["visual_field"]["delta_x"], r["visual_field"]["delta_y"])
            err = float(np.hypot(c[0] - gx, c[1] - gy))
            # Whole-frame centroid is dragged far from the true object by the clutter.
            self.assertGreater(err, 0.3)
            self.assertFalse(r["isolation"]["applied"])

    def test_depth_isolation_recovers_ground_truth(self):
        with tempfile.TemporaryDirectory() as d:
            ip, dp = _build(Path(d))
            r = verify_spatial_task(str(ip), task="bottom_left", depth_path=str(dp),
                                    table_z=0.0, isolate="depth_plane")
            gx, gy = _gt_centroid()
            c = (r["visual_field"]["delta_x"], r["visual_field"]["delta_y"])
            err = float(np.hypot(c[0] - gx, c[1] - gy))
            self.assertLess(err, 0.02)
            self.assertTrue(r["isolation"]["applied"])
            self.assertEqual(r["isolation"]["strategy"], "depth_plane")

    def test_isolation_default_is_backward_compatible(self):
        with tempfile.TemporaryDirectory() as d:
            ip, _ = _build(Path(d))
            r = verify_spatial_task(str(ip), task="bottom_left")
            self.assertIn(r["compliance"]["status"], {"PASS", "FAIL"})
            self.assertEqual(r["isolation"]["strategy"], "none")


class ParallaxTests(unittest.TestCase):
    def test_independent_agreement_yields_high_confidence(self):
        with tempfile.TemporaryDirectory() as d:
            ip, dp = _build(Path(d))
            obj = np.zeros((H, W), np.uint8)
            obj[OBJ_ROWS[0]:OBJ_ROWS[1], OBJ_COLS[0]:OBJ_COLS[1]] = 1
            out = verify_parallax(str(ip), task="bottom_left", depth_path=str(dp),
                                  table_z=0.0, rgb_isolate="explicit", object_mask=obj)
            p = out["parallax"]
            self.assertTrue(p["success"])
            self.assertEqual(p["independence"], 2)
            self.assertLess(p["omega"], 0.05)
            self.assertGreater(p["confidence"], 0.8)

    def test_disagreement_collapses_confidence(self):
        # RGB satisfied by a decoy; depth sees no object there → contested.
        rgb = Read("rgb_structure", "rgb", (-0.27, 0.28), True)
        depth = Read("depth_occupancy", "depth", (0.30, -0.30), False)
        v = fuse_reads([rgb, depth])
        self.assertFalse(v.success)
        self.assertGreater(v.omega, 0.3)
        self.assertLess(v.confidence, 0.2)

    def test_single_surface_confidence_is_capped(self):
        # One read, one surface: Ω measures nothing; confidence must not exceed 0.5.
        only = Read("rgb_structure", "rgb", (-0.27, 0.28), True)
        v = fuse_reads([only])
        self.assertEqual(v.independence, 1)
        self.assertLessEqual(v.confidence, 0.5)


if __name__ == "__main__":
    unittest.main()

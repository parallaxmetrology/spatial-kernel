import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from spatial_kernel.verify import report_to_json, verify_spatial_task
from spatial_kernel.robotics_er import build_robotics_er_prompt


class VtlSpatialTests(unittest.TestCase):
    def make_image(self, path: Path, position: str) -> None:
        img = np.full((512, 512, 3), 245, dtype=np.uint8)
        if position == "bottom_left":
            cv2.rectangle(img, (30, 310), (210, 480), (20, 20, 20), -1)
            cv2.line(img, (40, 330), (200, 460), (220, 220, 220), 4)
        elif position == "center":
            cv2.rectangle(img, (180, 180), (332, 332), (20, 20, 20), -1)
            cv2.line(img, (190, 190), (320, 320), (220, 220, 220), 4)
        else:
            raise ValueError(position)
        cv2.imwrite(str(path), img)

    def test_bottom_left_report_is_json_serializable(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "scene.png"
            self.make_image(path, "bottom_left")
            report = verify_spatial_task(str(path), task="bottom_left")
            parsed = json.loads(report_to_json(report))
            self.assertIn(parsed["compliance"]["status"], {"PASS", "FAIL"})
            self.assertIn("target_mass_ratio", parsed["region"])
            self.assertIn("sdi", parsed["visual_field"])

    def test_roi_changes_coordinate_frame(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "scene.png"
            self.make_image(path, "bottom_left")
            full = verify_spatial_task(str(path), task="bottom_left")
            roi = verify_spatial_task(str(path), task="center", roi=(0, 256, 256, 256))
            self.assertNotEqual(full["image"]["standardized_shape"], roi["image"]["standardized_shape"])
            self.assertTrue(roi["image"]["roi"]["applied"])

    def test_depth_optional_channel(self):
        with tempfile.TemporaryDirectory() as d:
            img_path = Path(d) / "scene.png"
            depth_path = Path(d) / "depth.npy"
            self.make_image(img_path, "center")
            depth = np.zeros((512, 512), dtype=np.float32)
            depth[180:332, 180:332] = 0.2
            np.save(depth_path, depth)
            report = verify_spatial_task(str(img_path), task="center", depth_path=str(depth_path), table_z=0.0)
            self.assertIn("depth_field", report)
            self.assertEqual(report["depth_field"]["valid"], 1)

    def test_robotics_er_prompt_uses_report(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "scene.png"
            self.make_image(path, "center")
            report = verify_spatial_task(str(path), task="center")
            prompt = build_robotics_er_prompt(report)
            self.assertIn("Do not decide pass/fail yourself", prompt)
            self.assertIn("REGION OCCUPANCY", prompt)


if __name__ == "__main__":
    unittest.main()

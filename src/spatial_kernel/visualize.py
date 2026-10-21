from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .core import KernelConfig, standardize_image_from_path
from .regions import normalized_to_px, region_mask
from .tasks import get_task_profile


def write_overlay(
    image_path: str,
    report: dict[str, Any],
    output_path: str,
    roi: tuple[int, int, int, int] | None = None,
    config: KernelConfig = KernelConfig(),
) -> None:
    img, _ = standardize_image_from_path(image_path, config=config, roi=roi)
    h, w = img.shape[:2]
    profile = get_task_profile(report["task"]["name"])
    target_region_mask = region_mask((h, w), profile.target_region)
    overlay = img.copy()
    color = np.array([40, 220, 255], dtype=np.uint8)
    overlay[target_region_mask] = (0.72 * overlay[target_region_mask] + 0.28 * color).astype(np.uint8)
    out = cv2.addWeighted(overlay, 0.78, img, 0.22, 0)

    measured = report["compliance"].get("measured_centroid_px")
    target = report["compliance"].get("target_px")
    if measured:
        cv2.circle(out, (int(measured[0]), int(measured[1])), 10, (0, 0, 255), -1)
        cv2.putText(out, "measured", (int(measured[0]) + 12, int(measured[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    if target:
        cv2.circle(out, (int(target[0]), int(target[1])), 10, (0, 255, 0), -1)
        cv2.putText(out, "target", (int(target[0]) + 12, int(target[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 160, 0), 2)
    if measured and target:
        cv2.arrowedLine(out, (int(measured[0]), int(measured[1])), (int(target[0]), int(target[1])), (255, 0, 0), 3, tipLength=0.08)

    status = report["compliance"]["status"]
    score = report["compliance"]["score"]
    label = f"{status} score={score:.3f}"
    cv2.rectangle(out, (12, 12), (min(w - 12, 360), 58), (0, 0, 0), -1)
    cv2.putText(out, label, (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite(output_path, out)

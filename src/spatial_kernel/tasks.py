from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TaskProfile:
    name: str
    target_region: str
    target: dict[str, float]
    weights: dict[str, float]
    threshold: float = 0.15
    min_target_mass_ratio: float | None = None
    max_leakage_ratio: float | None = None
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


TASKS: dict[str, TaskProfile] = {
    "bottom_left": TaskProfile(
        name="bottom_left",
        target_region="bottom_left",
        target={"delta_x": -0.40, "delta_y": 0.35, "r_v": 0.75, "x_p": 0.50, "sdi": 0.24},
        weights={"delta_x": 1.4, "delta_y": 1.2, "r_v": 0.7, "x_p": 0.35, "sdi": 0.45},
        threshold=0.18,
        min_target_mass_ratio=0.62,
        max_leakage_ratio=0.38,
        notes="Mass should be corner-loaded, not merely visually clustered near the lower-left.",
    ),
    "bottom_right": TaskProfile(
        name="bottom_right",
        target_region="bottom_right",
        target={"delta_x": 0.40, "delta_y": 0.35, "r_v": 0.75, "x_p": 0.50, "sdi": 0.24},
        weights={"delta_x": 1.4, "delta_y": 1.2, "r_v": 0.7, "x_p": 0.35, "sdi": 0.45},
        threshold=0.18,
        min_target_mass_ratio=0.62,
        max_leakage_ratio=0.38,
    ),
    "top_left": TaskProfile(
        name="top_left",
        target_region="top_left",
        target={"delta_x": -0.40, "delta_y": -0.35, "r_v": 0.75, "x_p": 0.50, "sdi": 0.24},
        weights={"delta_x": 1.4, "delta_y": 1.2, "r_v": 0.7, "x_p": 0.35, "sdi": 0.45},
        threshold=0.18,
        min_target_mass_ratio=0.62,
        max_leakage_ratio=0.38,
    ),
    "top_right": TaskProfile(
        name="top_right",
        target_region="top_right",
        target={"delta_x": 0.40, "delta_y": -0.35, "r_v": 0.75, "x_p": 0.50, "sdi": 0.24},
        weights={"delta_x": 1.4, "delta_y": 1.2, "r_v": 0.7, "x_p": 0.35, "sdi": 0.45},
        threshold=0.18,
        min_target_mass_ratio=0.62,
        max_leakage_ratio=0.38,
    ),
    "center": TaskProfile(
        name="center",
        target_region="center",
        target={"delta_x": 0.0, "delta_y": 0.0, "r_v": 0.70, "x_p": 0.30, "sdi": 0.18},
        weights={"delta_x": 1.3, "delta_y": 1.3, "r_v": 0.35, "x_p": 0.8, "sdi": 0.55},
        threshold=0.16,
        min_target_mass_ratio=0.50,
        max_leakage_ratio=0.50,
    ),
    "clear_region": TaskProfile(
        name="clear_region",
        target_region="full",
        target={"mass_fraction": 0.0, "r_v": 0.95},
        weights={"mass_fraction": 2.0, "r_v": 0.8},
        threshold=0.08,
        min_target_mass_ratio=None,
        max_leakage_ratio=None,
        notes="Target ROI should be visually empty or nearly empty.",
    ),
    "consolidate": TaskProfile(
        name="consolidate",
        target_region="center",
        target={"rho_r": 35.0, "sdi": 0.15, "mu": 0.20, "r_v": 0.72},
        weights={"rho_r": 0.012, "sdi": 1.2, "mu": 0.6, "r_v": 0.45},
        threshold=0.20,
        min_target_mass_ratio=0.50,
        max_leakage_ratio=0.50,
        notes="Mass should become a compact accessible group.",
    ),
    "spread": TaskProfile(
        name="spread",
        target_region="full",
        target={"sdi": 0.32, "rho_r": 12.0, "mu": 0.05},
        weights={"sdi": 1.2, "rho_r": 0.015, "mu": 0.5},
        threshold=0.18,
        notes="Mass should distribute across the available field.",
    ),
}


ALIASES = {
    "bottom-left": "bottom_left",
    "bottom left": "bottom_left",
    "lower_left": "bottom_left",
    "lower-left": "bottom_left",
    "lower left": "bottom_left",
    "bottom-right": "bottom_right",
    "bottom right": "bottom_right",
    "top-left": "top_left",
    "top left": "top_left",
    "top-right": "top_right",
    "top right": "top_right",
    "centered": "center",
    "centre": "center",
    "clear": "clear_region",
    "empty": "clear_region",
    "cluster": "consolidate",
    "compact": "consolidate",
    "separate": "spread",
    "distribute": "spread",
}


def get_task_profile(name: str) -> TaskProfile:
    key = ALIASES.get(name.strip().lower(), name.strip().lower())
    if key not in TASKS:
        known = ", ".join(sorted(TASKS))
        raise KeyError(f"Unknown task '{name}'. Known tasks: {known}")
    return TASKS[key]

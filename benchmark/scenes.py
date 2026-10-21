"""Labeled adversarial robot-outcome scenes with ground truth.

Synthetic, in the repo's null-protocol tradition (cf. the seismology synthetic FK
tests). Each scene is a manipulation *outcome* with a known correct verdict, drawn
from classes chosen to stress exactly where a geometry-blind judge must guess:

  clean_success    — object placed in the target region, standing above the table.   PASS
  wrong_place      — object placed in the WRONG region.                              FAIL
  clutter_success  — correct placement amid heavy background texture.                PASS
  decoy            — a FLAT printed photo of the object in the target region; looks  FAIL
                     correct in RGB, but nothing was actually placed (no geometry).
  occluded_success — correct placement, partially occluded, still above the plane.   PASS
  empty            — nothing placed at all.                                          FAIL

Every scene carries RGB + an aligned depth map + a segmenter-style object mask
(what a SAM2-class model would return from appearance). The decoy is the crux: its
mask and RGB say "object in region"; only depth reveals there is no object there.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

H = W = 512
TARGET_REGION = "bottom_left"  # rows[h/2:], cols[:w/2]
OBJ_H, OBJ_W = 140, 160


EXPECTED_OBJECT_HEIGHT = 0.20  # the real manipuland's graspable height (m, schematic)


@dataclass
class Scene:
    cls: str
    rgb: np.ndarray
    depth: np.ndarray
    object_mask: np.ndarray | None  # segmenter output (appearance-derived)
    gt_pass: bool
    # The TRUE graspable height at the target, as a gripper would feel it — a
    # physical fact independent of the depth CAMERA (which can fail on glass /
    # occlusion). 0.0 = nothing graspable there. This is the proprioception channel.
    true_grasp_height: float = 0.0


def _draw_object(img: np.ndarray, top: int, left: int, with_relief: bool = True) -> np.ndarray:
    """Draw a solid object; return its silhouette mask."""
    bot, right = top + OBJ_H, left + OBJ_W
    cv2.rectangle(img, (left, top), (right, bot), (25, 25, 25), -1)
    if with_relief:
        cv2.line(img, (left + 12, top + 16), (right - 12, bot - 16), (210, 210, 210), 3)
    mask = np.zeros((H, W), np.uint8)
    mask[top:bot, left:right] = 1
    return mask


def _clutter(img: np.ndarray, rng: np.random.Generator, n: int = 200) -> None:
    for _ in range(n):
        x = int(rng.integers(285, 500)); y = int(rng.integers(20, 235))
        s = int(rng.integers(2, 6)); sh = int(rng.integers(30, 90))
        cv2.rectangle(img, (x, y), (x + s, y + s), (sh, sh, sh), -1)


def make_scene(cls: str, rng: np.random.Generator) -> Scene:
    img = np.full((H, W, 3), 245, dtype=np.uint8)
    depth = np.zeros((H, W), dtype=np.float32)  # table plane at 0
    mask: np.ndarray | None = None

    # jittered placement inside the bottom-left quadrant
    bl_top = int(rng.integers(300, 340)); bl_left = int(rng.integers(35, 70))
    # wrong region: top-right quadrant
    tr_top = int(rng.integers(40, 80)); tr_left = int(rng.integers(300, 340))

    if cls == "clean_success":
        mask = _draw_object(img, bl_top, bl_left)
        depth[bl_top:bl_top + OBJ_H, bl_left:bl_left + OBJ_W] = 0.20

    elif cls == "wrong_place":
        mask = _draw_object(img, tr_top, tr_left)
        depth[tr_top:tr_top + OBJ_H, tr_left:tr_left + OBJ_W] = 0.20

    elif cls == "clutter_success":
        mask = _draw_object(img, bl_top, bl_left)
        depth[bl_top:bl_top + OBJ_H, bl_left:bl_left + OBJ_W] = 0.20
        _clutter(img, rng, n=240)

    elif cls == "decoy":
        # A flat printed photo: looks like the object in RGB (segmenter masks it),
        # but it is paint on the table — zero height. Nothing was placed.
        mask = _draw_object(img, bl_top, bl_left)
        # depth stays flat everywhere → no occupancy above the plane.

    elif cls == "occluded_success":
        mask = _draw_object(img, bl_top, bl_left)
        depth[bl_top:bl_top + OBJ_H, bl_left:bl_left + OBJ_W] = 0.20
        # occluder bar across part of the object (appearance partly hidden), but the
        # object still stands above the plane where visible.
        oc_y = bl_top + OBJ_H // 3
        cv2.rectangle(img, (bl_left - 10, oc_y), (bl_left + OBJ_W + 10, oc_y + 22), (150, 150, 150), -1)
        mask[oc_y:oc_y + 22, :] = 0  # segmenter loses the occluded strip

    elif cls == "empty":
        mask = None  # nothing to segment

    # ---- adversarial classes that break PARALLAX (the depth surface lies) ----

    elif cls == "transparent_success":
        # A glass/transparent object IS correctly placed (GT=PASS). RGB sees a faint
        # object (segmenter still masks it); the depth sensor sees THROUGH it and
        # reads table level. Depth says "nothing above the plane" → parallax will
        # disagree with RGB and false-FAIL a real success.
        faint = img.copy()
        cv2.rectangle(faint, (bl_left, bl_top), (bl_left + OBJ_W, bl_top + OBJ_H), (205, 205, 205), -1)
        cv2.rectangle(faint, (bl_left, bl_top), (bl_left + OBJ_W, bl_top + OBJ_H), (180, 180, 180), 2)
        img = faint
        mask = np.zeros((H, W), np.uint8)
        mask[bl_top:bl_top + OBJ_H, bl_left:bl_left + OBJ_W] = 1
        # depth stays flat (sensor sees through glass)

    elif cls == "depth_dropout_success":
        # Object correctly placed (GT=PASS) but occluded from the depth camera's
        # viewpoint → depth has a dropout hole (NaN) exactly where the object is.
        mask = _draw_object(img, bl_top, bl_left)
        depth[:] = 0.0
        depth[bl_top:bl_top + OBJ_H, bl_left:bl_left + OBJ_W] = np.nan

    elif cls == "depth_noise_success":
        # Object correctly placed (GT=PASS) but the depth map is noisy: Gaussian
        # noise plus random above-threshold speckle elsewhere that can drag the
        # occupancy centroid out of the target region.
        mask = _draw_object(img, bl_top, bl_left)
        depth[bl_top:bl_top + OBJ_H, bl_left:bl_left + OBJ_W] = 0.20
        depth = depth + rng.normal(0.0, 0.04, size=depth.shape).astype(np.float32)
        speck = rng.random(depth.shape) < 0.01
        depth[speck] = 0.25  # phantom occupancy scattered across the frame

    elif cls == "raised_decoy":
        # A printed photo of the object placed on a low box in the target region.
        # RGB shows the object (segmenter masks it) AND depth now sees raised mass —
        # so RGB and depth AGREE it is there. But nothing manipulable was placed:
        # GT=FAIL. Parallax will be co-fooled and false-PASS. Only a third
        # independent surface (grasp/proprioception) could catch this.
        mask = _draw_object(img, bl_top, bl_left)
        depth[bl_top:bl_top + OBJ_H, bl_left:bl_left + OBJ_W] = 0.06  # the box

    else:
        raise ValueError(f"unknown class {cls}")

    gt_pass = cls in {
        "clean_success", "clutter_success", "occluded_success",
        "transparent_success", "depth_dropout_success", "depth_noise_success",
    }
    # True graspable height at the target (what a gripper feels), independent of the
    # depth camera. Real object → expected height; flat decoy / empty / object-elsewhere
    # → nothing; raised_decoy → a box, present but the WRONG profile.
    grasp = {
        "clean_success": EXPECTED_OBJECT_HEIGHT,
        "clutter_success": EXPECTED_OBJECT_HEIGHT,
        "occluded_success": EXPECTED_OBJECT_HEIGHT,
        "transparent_success": EXPECTED_OBJECT_HEIGHT,   # glass IS graspable
        "depth_dropout_success": EXPECTED_OBJECT_HEIGHT,  # occluded from camera, not from gripper
        "depth_noise_success": EXPECTED_OBJECT_HEIGHT,
        "wrong_place": 0.0,      # nothing at the TARGET
        "decoy": 0.0,            # paper on the table — no graspable mass
        "empty": 0.0,
        "raised_decoy": 0.06,    # the box: present, but wrong profile vs the object
    }.get(cls, 0.0)
    return Scene(cls=cls, rgb=img, depth=depth, object_mask=mask, gt_pass=gt_pass,
                true_grasp_height=grasp)


CLASSES = [
    "clean_success", "wrong_place", "clutter_success", "decoy", "occluded_success", "empty",
    # adversarial-against-parallax (v2):
    "transparent_success", "depth_dropout_success", "depth_noise_success", "raised_decoy",
]


def generate(per_class: int = 6, seed: int = 7) -> list[Scene]:
    rng = np.random.default_rng(seed)
    scenes: list[Scene] = []
    for cls in CLASSES:
        for _ in range(per_class):
            scenes.append(make_scene(cls, rng))
    return scenes

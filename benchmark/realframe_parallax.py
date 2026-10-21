#!/usr/bin/env python3
"""
Real-frame two-view parallax analyzer.

Reads paired photos (a scene shot from two ~10 cm-offset camera positions,
"Dot1" and "Dot2"), and asks the only question that separates a real 3D object
from a flat printed decoy: does the object shift MORE than the table plane
between the two views?

- Table plane: measured from the two black fiducial dots taped on the table.
- Object: the salient (colored / high-contrast) blob that is not a dot.
- relative_parallax = object_shift - table_shift   (pixels, at the analysis scale)
    ~0        -> object lies in the table plane  -> FLAT / at-plane
    large     -> object stands above the plane   -> RAISED / real

Ground truth is read from the filename (the class name), so no separate
labeling step is needed. This is the real-pixel counterpart to the synthetic
benchmark/test_stereo.py.
"""
import os, re, glob, json
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

import sys
_BASE = os.path.dirname(__file__)
CUPS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_BASE, "..", "data", "cups2")
OUT  = os.path.join(_BASE, "..", "results", os.path.basename(os.path.normpath(CUPS)).lower())

# class -> (has_object, expected_surface, note). expected_surface: 'raised'|'flat'|None
CLASSES = {
    "bare table":                 (False, None,     "empty — nothing placed"),
    # real 3D objects (should read RAISED)
    "real mug":                   (True,  "raised", "real mug, in place"),
    "real mug right":             (True,  "raised", "real mug, wrong place"),
    "paper cup":                  (True,  "raised", "real disposable paper cup"),
    "clear glass":                (True,  "raised", "real glass (transparent stressor)"),
    "scotch roll":                (True,  "raised", "real 3D tape roll"),
    "scotch box":                 (True,  "raised", "real 3D box"),
    "scotch box side":            (True,  "raised", "real box standing on end"),
    # flat prints / cutouts (should read FLAT — the decoys)
    "flat green l":               (True,  "flat",   "flat paper L (clean control)"),
    "cut out mug":                (True,  "flat",   "cutout print of mug, flat"),
    "cut out paper cup":          (True,  "flat",   "cutout print of paper cup, flat"),
    "cut out scotch roll":        (True,  "flat",   "cutout print of tape roll, flat"),
    "full page mug":              (True,  "flat",   "full-page print of mug, flat"),
    "full page paper cup":        (True,  "flat",   "full-page print of paper cup, flat"),
    "full page scotch roll":      (True,  "flat",   "full-page print of tape roll, flat"),
    "scotch box flat":            (True,  "flat",   "flat print of scotch box"),
    "scotch box fake 3d":         (True,  "flat",   "flattened box, flat but looks 3D"),
    # flat print raised on a box (should read RAISED — the 2-surface trap)
    "cut out mug on the box":     (True,  "raised", "flat mug print on a box — trap"),
    "scotch box flat on the box": (True,  "raised", "flat box print on a box — trap"),
}

def scene_key(fname):
    base = os.path.splitext(os.path.basename(fname))[0].lower()
    base = re.sub(r"\s*dot\s*[12]\s*$", "", base).strip()
    return base

def class_of(key):
    """Map a scene key (e.g. 'real mug 02', 'real mug right 01') to its class by
    longest matching prefix, so numbered multi-sample files just work."""
    for prefix in sorted(CLASSES, key=len, reverse=True):
        if key == prefix or key.startswith(prefix + " "):
            return prefix, CLASSES[prefix]
    return None, (True, None, "unrecognized class")

from spatial_kernel.twoview import (  # noqa: F401  (re-exported for placement_precision)
    SCALE_W, analyze_pair, components, find_dots, find_object, load, saturation, verdict)

def draw_overlay(pack, rel, path):
    r1, r2, d1, d2, o1, o2 = pack
    H, W = r1.shape[:2]
    canvas = Image.new("RGB", (W * 2 + 30, H), (18, 22, 28))
    for k, (rgb, dd, oo, tag) in enumerate([(r1, d1, o1, "Dot1"), (r2, d2, o2, "Dot2")]):
        img = Image.fromarray(rgb.astype(np.uint8))
        dr = ImageDraw.Draw(img)
        if dd:
            for d in dd[1:]:
                dr.ellipse([d["cx"]-14, d["cy"]-14, d["cx"]+14, d["cy"]+14], outline=(70,150,255), width=4)
            dr.line([dd[0], 0, dd[0], H], fill=(70,150,255), width=2)
        if oo:
            dr.ellipse([oo["cx"]-16, oo["cy"]-16, oo["cx"]+16, oo["cy"]+16], outline=(232,131,58), width=5)
            dr.line([oo["cx"], 0, oo["cx"], H], fill=(232,131,58), width=2)
        dr.text((14, 12), tag, fill=(255,255,255))
        canvas.paste(img, (k * (W + 30), 0))
    ImageDraw.Draw(canvas).text((14, H-24),
        f"relative parallax = {rel} px  (orange=object, blue=table plane)" if rel is not None else "no object",
        fill=(255,255,255))
    canvas.save(path)

def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(CUPS, "*.jpeg")) + glob.glob(os.path.join(CUPS, "*.jpg")))
    scenes = {}
    for f in files:
        scenes.setdefault(scene_key(f), {})
        if re.search(r"dot\s*1", os.path.basename(f).lower()): scenes[scene_key(f)]["1"] = f
        elif re.search(r"dot\s*2", os.path.basename(f).lower()): scenes[scene_key(f)]["2"] = f

    CAP = 90.0     # |rel| beyond this (at analysis scale) = a mis-track, not a real verdict
    THRESH = 15.0  # px; matches batch-1's auto-calibrated 15.1

    rows, packs = [], {}
    for key, pair in sorted(scenes.items()):
        if "1" not in pair or "2" not in pair:
            continue
        res, pack = analyze_pair(pair["1"], pair["2"])
        packs[key] = pack
        rows.append((key, res))

    # classify every scene: label, correct(bool), graded(bool), detectfail(bool)
    per = {}
    out_json = dict(threshold_px=THRESH, cap_px=CAP, scenes={})
    for key, res in rows:
        exp = class_of(key)[1][1]
        rel = res.get("relative_parallax_px")
        detectfail = rel is not None and abs(rel) > CAP
        if exp is None:                                   # bare table: success = detect nothing
            label = "no-object" if rel is None else ("detect-fail" if detectfail else verdict(rel, THRESH))
            correct, graded = (rel is None), True
        elif rel is None or detectfail:                   # object expected but not cleanly found
            label = "detect-fail" if detectfail else "MISSED"
            correct, graded = False, False                # excluded from discrimination accuracy
        else:
            got = "raised" if abs(rel) >= THRESH else "flat"
            label, correct, graded = got.upper(), (got == exp), True
        per[key] = dict(exp=exp, rel=rel, label=label, correct=correct, graded=graded, detectfail=detectfail)
        draw_overlay(packs[key], rel, os.path.join(OUT, f"{key.replace(' ', '_')}_overlay.png"))
        ts = res.get("table_shift_px")
        out_json["scenes"][key] = dict(
            relative_parallax_px=(None if rel is None else float(rel)), expected=exp, verdict=label,
            correct=bool(correct), detect_fail=bool(detectfail),
            table_shift_px=(None if ts is None else float(ts)))

    # per-class rollup
    classes = {}
    for key, p in per.items():
        cls = class_of(key)[0] or key
        classes.setdefault(cls, []).append(p)

    print("=" * 84)
    print(f"REAL-FRAME TWO-VIEW PARALLAX  ·  {os.path.basename(os.path.normpath(CUPS))}  ·  "
          f"{len(rows)} scenes  ·  threshold {THRESH}px")
    print("=" * 84)
    print(f"{'class':<28}{'exp':<7}{'n':>3}{'det':>5}{'mean|rel|':>11}{'correct':>9}")
    print("-" * 84)
    tot_graded = tot_correct = tot_detfail = 0
    for cls in sorted(classes):
        ps = classes[cls]
        exp = ps[0]["exp"] or "none"
        det = [p for p in ps if not p["detectfail"] and p["rel"] is not None]
        detfail = sum(1 for p in ps if p["detectfail"] or (p["exp"] is not None and p["rel"] is None))
        graded = [p for p in ps if p["graded"]]
        corr = sum(1 for p in graded if p["correct"])
        meanrel = (sum(abs(p["rel"]) for p in det) / len(det)) if det else float("nan")
        tot_graded += len(graded); tot_correct += corr; tot_detfail += detfail
        print(f"{cls:<28}{exp:<7}{len(ps):>3}{len(ps)-detfail:>5}{meanrel:>11.1f}{corr:>4}/{len(graded):<4}")
    print("-" * 84)
    print(f"discrimination accuracy (scenes cleanly detected): {tot_correct}/{tot_graded}")
    print(f"detection failures (mis-track or object not found): {tot_detfail}")

    flagged = [(k, p) for k, p in sorted(per.items())
               if p["detectfail"] or (p["graded"] and not p["correct"]) or (p["exp"] is not None and p["rel"] is None)]
    if flagged:
        print("\nflagged scenes (detect-fail / miss / wrong):")
        for k, p in flagged:
            relstr = "" if p["rel"] is None else f"{p['rel']:+.1f}px"
            print(f"  {k:<30}{relstr:>10}  {p['label']}  (exp {p['exp']})")
    print(f"\noverlays: {os.path.relpath(OUT)}")
    with open(os.path.join(OUT, "realframe_results.json"), "w") as fh:
        json.dump(out_json, fh, indent=2)

if __name__ == "__main__":
    main()

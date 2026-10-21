#!/usr/bin/env python3
"""
Placement-precision head-to-head: the instrument MEASURES where a real object sits
relative to a target; Gemini APPROXIMATES it.

Each single photo contains:
  - two black fiducial dots (~10 cm apart)  -> pixel->cm scale
  - a GREEN target marker                   -> where the object should go (T)
  - the real object (e.g. a mug) at some offset from T (O)

The instrument reports the exact gap |O - T| (cm and normalized) and the correction
vector to fix it. Gemini is asked (a) PASS/FAIL: "was it placed on the target?" and
(b) to point to the object's center. We then show where Gemini's coarse PASS/FAIL
boundary sits vs. the measured gap (it false-passes off-target placements), and how
far its point drifts from the measured position.

This is the "off by millimeters / looks done but isn't" failure — with REAL objects,
no constructed decoys. Usage: python3 -m benchmark.placement_precision [folder]
"""
import os, re, glob, json, io, sys, math
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from benchmark.realframe_parallax import load, find_dots, find_object, components

FOLDER = sys.argv[1] if len(sys.argv) > 1 else "placement"
BASE = os.path.join(os.path.dirname(__file__), "..")
SRC = os.path.join(BASE, "data", FOLDER.lower())
OUT = os.path.join(BASE, "results", FOLDER.lower())
MODEL = "gemini-robotics-er-1.6-preview"
DOT_SPACING_CM = 10.0     # real distance between the two fiducial dots
TOL_CM = 2.0              # success tolerance: object center within this of target
OBJECT_NOUN = "mug"

def find_green(rgb, dots):
    """Green target marker. Returns (blob dict or None, green pixel mask). The mask is
    the actual green pixels — so we can blank ONLY the green, preserving a cup that sits
    inside/over it (a big target with a small cup near dead-on)."""
    H, W = rgb.shape[:2]
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    green = (G > R + 20) & (G > B + 20) & (G > 60)
    if dots:
        for d in dots[1:]:
            yy, xx, r = int(d["cy"]), int(d["cx"]), int(3 * max(d["w"], d["h"]))
            green[max(0, yy - r):yy + r, max(0, xx - r):xx + r] = False
    green = ndimage.binary_opening(green, iterations=2)
    comps = components(green, min_area=int(0.0002 * H * W))
    if not comps:
        return None, green
    blob = max(comps, key=lambda c: c["area"])
    return blob, green

def fit_green_center(rgb, greenmask, min_pts=40):
    """Recover the TRUE center of the (circular) green target even when the cup occludes
    part of it: fit a circle to the green pixels that border the bright background — the
    real green-to-table arc — ignoring the green-to-cup edge. Returns (cx, cy) or None."""
    if greenmask is None or not greenmask.any():
        return None
    bright = rgb.mean(2) > 150
    arc = greenmask & ndimage.binary_dilation(bright, iterations=2)  # green edge against table only
    ys, xs = np.where(arc)
    if len(xs) < min_pts:
        return None
    x = xs.astype(float); y = ys.astype(float)          # algebraic (Kåsa) circle fit
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    try:
        sol, *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    except Exception:
        return None
    return float(sol[0]), float(sol[1])

def mask_out(rgb, greenmask):
    """Blank ONLY the green pixels (dilated a little) to table gray, so find_object keeps
    a cup sitting inside the green ring."""
    if greenmask is None or not greenmask.any():
        return rgb
    out = rgb.copy()
    out[ndimage.binary_dilation(greenmask, iterations=6)] = float(np.median(rgb))
    return out

def analyze(path):
    rgb, s = load(path)
    H, W = rgb.shape[:2]
    dots = find_dots(rgb)
    green, gmask = find_green(rgb, dots)
    obj = find_object(mask_out(rgb, gmask), dots)   # object, with only green pixels painted out
    res = dict(path=os.path.relpath(path, BASE), W=W, H=H)
    if not dots:
        res["error"] = "dots not found"; return res, rgb, None, green, obj
    d1, d2 = dots[1], dots[2]
    dot_px = math.hypot(d1["cx"] - d2["cx"], d1["cy"] - d2["cy"])
    px_per_cm = dot_px / DOT_SPACING_CM
    res["px_per_cm"] = px_per_cm
    if green is None or obj is None:
        res["error"] = f"missing {'green target' if green is None else 'object'}"
        return res, rgb, dots, green, obj
    fit = fit_green_center(rgb, gmask)                       # true circle center, occlusion-robust
    T = np.array(fit if fit else [green["cx"], green["cy"]])
    O = np.array([obj["cx"], obj["cy"]])
    gap_px = float(np.hypot(*(O - T)))
    res.update(target_px=T.tolist(), object_px=O.tolist(),
               gap_px=round(gap_px, 1), gap_cm=round(gap_px / px_per_cm, 2),
               gap_norm=round(gap_px / W, 3),
               correction_norm=[round(float((T - O)[0] / W), 3), round(float((T - O)[1] / W), 3)],
               instrument=("PASS" if gap_px / px_per_cm <= TOL_CM else "FAIL"))
    return res, rgb, dots, green, obj

# ---------- Gemini ----------
def _png(rgb):
    im = Image.fromarray(rgb.astype(np.uint8)); buf = io.BytesIO(); im.save(buf, "PNG"); return buf.getvalue()

def gemini(rgb, api_key, W, H):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)
    def ask(prompt):
        contents = [types.Content(role="user", parts=[
            types.Part.from_bytes(data=_png(rgb), mime_type="image/png"),
            types.Part.from_text(text=prompt)])]
        cfg = types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_level="LOW"))
        t = ""
        for ch in client.models.generate_content_stream(model=MODEL, contents=contents, config=cfg):
            if ch.text: t += ch.text
        return t.strip()
    verdict_txt = ask(
        f"A top-down photo of a tabletop. TASK: place a {OBJECT_NOUN} centered on the GREEN "
        f"target marker. Did it succeed? Count PASS only if the {OBJECT_NOUN} is resting ON the "
        f"green marker; FAIL if it is off to the side. First line exactly 'VERDICT: PASS' or "
        f"'VERDICT: FAIL', then one short sentence.")
    m = re.search(r"VERDICT:\s*(PASS|FAIL)", verdict_txt, re.I)
    verdict = (m.group(1).upper() == "PASS") if m else None

    def point(thing):
        txt = ask(f"Point to the center of the {thing}. Respond with ONLY a JSON array [x, y] in "
                  f"normalized coordinates from 0 to 1000 (x horizontal, left=0; y vertical, top=0). "
                  f"No other text.")
        n = re.findall(r"-?\d+\.?\d*", txt)
        return ([float(n[0]) / 1000 * W, float(n[1]) / 1000 * H] if len(n) >= 2 else None), txt

    obj_pt, obj_raw = point(OBJECT_NOUN)
    tgt_pt, tgt_raw = point("green target marker")
    return dict(gemini_pass=verdict, gemini_point=obj_pt, gemini_target_point=tgt_pt,
                verdict_raw=verdict_txt, obj_point_raw=obj_raw, tgt_point_raw=tgt_raw)

def overlay(rgb, dots, green, obj, res, gpt, path):
    img = Image.fromarray(rgb.astype(np.uint8)); dr = ImageDraw.Draw(img)
    if dots:
        for d in dots[1:]:
            dr.ellipse([d["cx"]-12, d["cy"]-12, d["cx"]+12, d["cy"]+12], outline=(70,150,255), width=4)
    if green:
        dr.ellipse([green["cx"]-16, green["cy"]-16, green["cx"]+16, green["cy"]+16], outline=(60,200,110), width=5)
    if obj:
        dr.ellipse([obj["cx"]-16, obj["cy"]-16, obj["cx"]+16, obj["cy"]+16], outline=(232,131,58), width=5)
    if green and obj:
        dr.line([obj["cx"], obj["cy"], green["cx"], green["cy"]], fill=(232,131,58), width=3)  # correction vector
    if gpt and gpt.get("gemini_point"):
        gx, gy = gpt["gemini_point"]
        dr.line([gx-14, gy, gx+14, gy], fill=(255,80,80), width=3); dr.line([gx, gy-14, gx, gy+14], fill=(255,80,80), width=3)
    cap = f"gap {res.get('gap_cm','?')}cm  instrument={res.get('instrument','?')}"
    if gpt: cap += f"  Gemini={'PASS' if gpt.get('gemini_pass') else 'FAIL' if gpt.get('gemini_pass') is False else '?'}"
    dr.text((14, 14), cap, fill=(255,255,255))
    img.save(path)

def main():
    api_key = None
    envp = os.path.join(BASE, ".env")
    if os.path.exists(envp):
        for line in open(envp):
            if line.startswith("GEMINI_API_KEY"): api_key = line.split("=", 1)[1].strip()
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SRC, "*.jpeg")) + glob.glob(os.path.join(SRC, "*.jpg")))
    if not files:
        print(f"no images in {SRC}"); return

    print("=" * 96)
    print(f"PLACEMENT PRECISION  ·  instrument measures vs Gemini approximates  ·  {FOLDER}  ·  tol {TOL_CM}cm")
    print("=" * 96)
    print(f"{'scene':<20}{'meas_cm':>8}{'instr':>7}{'Gemini':>8}{'G_impl_cm':>10}{'self-incon':>11}{'pt_err_cm':>10}")
    print("-" * 96)
    out = {}
    n_gfp = n_selfinc = 0
    for f in files:
        key = os.path.splitext(os.path.basename(f))[0]
        res, rgb, dots, green, obj = analyze(f)
        gpt = None
        if "error" not in res and api_key:
            gpt = gemini(rgb, api_key, res["W"], res["H"])
        overlay(rgb, dots, green, obj, res, gpt, os.path.join(OUT, key.replace(" ", "_") + "_place.png"))
        if "error" in res:
            print(f"{key:<20}{'—':>8}{'—':>7}{'—':>8}   {res['error']}"); out[key] = res; continue
        gp = gpt.get("gemini_pass") if gpt else None
        gpass = "PASS" if gp else ("FAIL" if gp is False else "?")
        false_pass = (gp is True and res["instrument"] == "FAIL")
        n_gfp += int(false_pass)
        ppc = res["px_per_cm"]
        # Gemini's own implied gap (its object point vs its target point) and its point error vs our measurement
        impl_cm = selfinc = pterr = None
        if gpt and gpt.get("gemini_point") and gpt.get("gemini_target_point"):
            gop, gtp = np.array(gpt["gemini_point"]), np.array(gpt["gemini_target_point"])
            impl_cm = float(np.hypot(*(gop - gtp)) / ppc)
            selfinc = (gp is True and impl_cm > TOL_CM)      # passed, yet its OWN points show a miss
            n_selfinc += int(selfinc)
            pterr = float(np.hypot(*(gop - np.array(res["object_px"]))) / ppc)
        res["gemini_implied_gap_cm"] = None if impl_cm is None else round(impl_cm, 2)
        res["gemini_self_inconsistent"] = bool(selfinc) if selfinc is not None else None
        print(f"{key:<20}{res['gap_cm']:>8.2f}{res['instrument']:>7}{gpass:>8}"
              f"{('' if impl_cm is None else f'{impl_cm:.2f}'):>10}"
              f"{('YES' if selfinc else '-' if selfinc is not None else '?'):>11}"
              f"{('' if pterr is None else f'{pterr:.2f}'):>10}")
        out[key] = {**res, **(gpt or {})}
    print("-" * 96)
    print(f"Gemini false-passes vs the instrument's measured gap: {n_gfp}")
    print(f"Gemini SELF-inconsistent (PASS while its own two points imply a > {TOL_CM}cm miss): {n_selfinc}")
    print(f"overlays + json -> {os.path.relpath(OUT)}")
    json.dump(out, open(os.path.join(OUT, "placement_precision.json"), "w"), indent=2, default=float)

if __name__ == "__main__":
    main()

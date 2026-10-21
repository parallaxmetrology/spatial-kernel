"""
Two-view parallax check on ordinary photos: is the object really standing on the
table, or is it a flat print?

Two photos of one scene from camera positions ~10 cm apart ("Dot1", "Dot2"), with two
black fiducial dots taped on the table. The table plane's shift comes from the dots;
the object's shift from its edge/saturation blob. relative parallax = object shift -
table shift (pixels at a 1400-px analysis width): about 0 means the object lies in the
table plane (FLAT); a large value means it stands above it (RAISED).

The functions below are the research code (benchmark/realframe_parallax.py) unchanged;
`check_pair` adds the published threshold and mis-track cap.
"""
import numpy as np
from PIL import Image
from scipy import ndimage

SCALE_W = 1400  # analysis width; full res is 4032
THRESH = 15.0   # px; |relative parallax| at or above this reads RAISED
CAP = 90.0      # px; beyond this the object was mis-tracked, not measured


def load(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = SCALE_W / w
    im = im.resize((SCALE_W, int(round(h * s))))
    return np.asarray(im).astype(np.float32), s

def saturation(rgb):
    mx = rgb.max(2); mn = rgb.min(2)
    return (mx - mn) / (mx + 1e-6)

def components(mask, min_area):
    lbl, n = ndimage.label(mask)
    out = []
    for i in range(1, n + 1):
        ys, xs = np.where(lbl == i)
        if len(xs) < min_area:
            continue
        w = xs.max() - xs.min() + 1
        h = ys.max() - ys.min() + 1
        out.append(dict(area=len(xs), cx=xs.mean(), cy=ys.mean(),
                        w=w, h=h, fill=len(xs) / (w * h), aspect=w / max(h, 1)))
    return out

def find_dots(rgb):
    """Two near-black circular fiducials, lower region, similar size."""
    H, W = rgb.shape[:2]
    dark = rgb.mean(2) < 70
    cands = [c for c in components(dark, min_area=40)
             if c["cy"] > 0.45 * H                      # lower region
             and 0.55 < c["fill"] < 0.98                # roughly filled circle
             and 0.6 < c["aspect"] < 1.7
             and c["area"] < 0.01 * H * W]              # small
    if len(cands) < 2:
        return None
    # choose the pair most similar in area with a real horizontal gap, similar y
    best, bestscore = None, 1e18
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            a, b = cands[i], cands[j]
            dy = abs(a["cy"] - b["cy"])
            dx = abs(a["cx"] - b["cx"])
            if dx < 0.04 * W:            # must be separated horizontally
                continue
            score = abs(a["area"] - b["area"]) / max(a["area"], b["area"]) + dy / H
            if score < bestscore:
                bestscore, best = score, (a, b)
    if best is None:
        return None
    return np.mean([best[0]["cx"], best[1]["cx"]]), best[0], best[1]

def find_object(rgb, dots):
    """Locate the object by its EDGES (robust to low contrast and smooth lighting
    gradients) OR strong saturation. A real object has sharp boundaries; a smooth
    vignette on a bare table does not — so this returns None on an empty table."""
    H, W = rgb.shape[:2]
    exclude = np.zeros((H, W), bool)
    if dots:
        for d in dots[1:]:
            yy, xx = int(d["cy"]), int(d["cx"])
            r = int(3 * max(d["w"], d["h"]))
            exclude[max(0, yy - r):yy + r, max(0, xx - r):xx + r] = True

    gray = rgb.mean(2)
    sat = saturation(rgb)
    edge = ndimage.gaussian_gradient_magnitude(gray, sigma=2.0)
    e_thr = edge.mean() + 3.0 * edge.std()             # only genuinely sharp boundaries
    mask = ((edge > e_thr) | (sat > 0.30)) & ~exclude
    mask = ndimage.binary_closing(mask, iterations=4)  # fill the object interior from its rim
    mask = ndimage.binary_opening(mask, iterations=2)  # drop speckle / dust / smooth noise
    comps = components(mask, min_area=int(0.0008 * H * W))
    if not comps:
        return None                                    # bare table lands here
    obj = max(comps, key=lambda c: c["area"])
    if obj["area"] > 0.5 * H * W:                      # whole-frame blob = not an object
        return None
    obj["mode"] = "edge"
    return obj

def analyze_pair(f1, f2):
    r1, s1 = load(f1); r2, s2 = load(f2)
    d1 = find_dots(r1); d2 = find_dots(r2)
    o1 = find_object(r1, d1); o2 = find_object(r2, d2)
    res = dict(ok=False)
    if not (d1 and d2):
        res["error"] = "dots not found in both views"
        return res, (r1, r2, d1, d2, o1, o2)
    table_shift = d2[0] - d1[0]
    res["table_shift_px"] = round(table_shift, 1)
    if o1 and o2:
        object_shift = o2["cx"] - o1["cx"]
        rel = object_shift - table_shift
        res.update(ok=True,
                   object_shift_px=round(object_shift, 1),
                   relative_parallax_px=round(rel, 1),
                   object_mode=o1.get("mode"))
    else:
        res["ok"] = True
        res["relative_parallax_px"] = None  # no object (e.g. bare table)
    return res, (r1, r2, d1, d2, o1, o2)

def verdict(rel, thresh):
    if rel is None:
        return "no-object"
    return "RAISED (above plane)" if abs(rel) >= thresh else "FLAT (at plane)"


def check_pair(view1, view2, thresh=THRESH, cap=CAP):
    """Verdict for one scene from two photos: RAISED, FLAT, no-object or detect-fail."""
    res, _ = analyze_pair(view1, view2)
    if "error" in res:
        return dict(res, verdict="error")
    rel = res.get("relative_parallax_px")
    if rel is None:
        label = "no-object"
    elif abs(rel) > cap:
        label = "detect-fail"
    else:
        label = "RAISED" if abs(rel) >= thresh else "FLAT"
    return dict(res, verdict=label, threshold_px=thresh, cap_px=cap)

"""Build the images and data for the Real or printed? game (docs/).

For each scene Gemini judged (results/recorded/cups2/realframe_vlm.json), both views
are aligned on the table's two fiducial dots (shift, rotation and scale), so that flipping between them leaves the
table still: a flat print stays put, a real object jumps. Needs the photos
(python tools/fetch_photos.py).
"""
import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from spatial_kernel.twoview import find_dots, load

ROOT = Path(__file__).resolve().parents[1]
PHOTOS = ROOT / "data" / "cups2"
OUT = ROOT / "docs" / "scenes"
WIDTH = 960

KIND = {
    "cut out mug": "a cut-out print of a mug",
    "full page mug": "a full-page print of a mug",
    "full page paper cup": "a full-page print of a paper cup",
    "scotch box fake 3d": "a flattened box with printed 3D shading",
    "scotch box flat": "a flat print of a Scotch tape box",
    "real mug": "a real mug",
    "paper cup": "a real paper cup",
    "scotch box side": "a real Scotch tape box on its side",
    "bare table": "an empty table",
}


def align(r1, r2, d1, d2, margin=0.04):
    """Warp view 2 onto view 1 with the similarity (shift, rotation, scale) that maps
    its two dots onto view 1's dots; crop a margin so both frames are fully covered."""
    def pts(d):
        p = sorted(((x["cx"], x["cy"]) for x in d[1:]))
        return complex(*p[0]), complex(*p[1])
    a1, b1 = pts(d1)
    a2, b2 = pts(d2)
    s = (b1 - a1) / (b2 - a2)
    t = a1 - s * a2
    m = np.float32([[s.real, -s.imag, t.real], [s.imag, s.real, t.imag]])
    h, w = r1.shape[:2]
    warped = cv2.warpAffine(r2.astype(np.uint8), m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    cx, cy = int(w * margin), int(h * margin)
    return r1[cy:h - cy, cx:w - cx], warped[cy:h - cy, cx:w - cx]


def main():
    vlm = json.loads((ROOT / "results/recorded/cups2/realframe_vlm.json").read_text())
    par = json.loads((ROOT / "results/recorded/cups2/realframe_results.json").read_text())["scenes"]
    files = {p.name.lower(): p for p in PHOTOS.iterdir()}
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for key, g in vlm.items():
        kind = re.sub(r" \d+$", "", key)
        r1, _ = load(files[f"{key} dot1.jpeg"])
        r2, _ = load(files[f"{key} dot2.jpeg"])
        d1, d2 = find_dots(r1), find_dots(r2)
        a, b = align(r1, r2, d1, d2)
        slug = key.replace(" ", "-")
        for i, arr in ((1, a), (2, b)):
            im = Image.fromarray(arr.astype(np.uint8))
            im = im.resize((WIDTH, round(im.height * WIDTH / im.width)), Image.LANCZOS)
            im.save(OUT / f"{slug}-{i}.jpg", "JPEG", quality=82, optimize=True)
        p = par[key]
        rows.append(dict(id=slug, noun=g["noun"], real=g["real_object"], kind=KIND[kind],
                         gemini_pass=g["gemini_pass"], gemini_text=g["raw"].strip(),
                         verdict=p["verdict"], rel_px=p["relative_parallax_px"]))
    (ROOT / "docs" / "game-data.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    print(len(rows), "scenes ->", OUT)


if __name__ == "__main__":
    main()

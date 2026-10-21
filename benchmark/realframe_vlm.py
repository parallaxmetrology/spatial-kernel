#!/usr/bin/env python3
"""
Real-pixel head-to-head: Gemini Robotics-ER (RGB-only judge) vs. the deterministic
two-view parallax instrument, across object types.

The VLM sees ONE real RGB frame (Dot1) per scene and a NEUTRAL, object-named presence
prompt — it is NOT told decoys exist, NOT given depth or the parallax verdict. We then
line its verdict up against ground truth (from the filename) and against the parallax
instrument's verdict (loaded from results/<folder>/realframe_results.json).

Curated to the strong flat decoys + real controls + empty — the cases where geometry
CAN disagree with appearance. Usage: python3 -m benchmark.realframe_vlm [folder]
"""
import os, re, glob, json, io, sys
from PIL import Image

FOLDER = sys.argv[1] if len(sys.argv) > 1 else "cups2"
BASE = os.path.join(os.path.dirname(__file__), "..")
CUPS = os.path.join(BASE, "data", FOLDER.lower())
PARALLAX_JSON = os.path.join(BASE, "results", FOLDER.lower(), "realframe_results.json")
MODEL = "gemini-robotics-er-1.6-preview"
MAXW = 1024
N_PER_CLASS = 3

# class -> (object noun for the prompt, is a REAL 3D object present?)
CLASS = {
    "cut out mug":          ("mug", False),
    "full page mug":        ("mug", False),
    "full page paper cup":  ("cup", False),
    "scotch box fake 3d":   ("box of Scotch tape", False),
    "scotch box flat":      ("box of Scotch tape", False),
    "real mug":             ("mug", True),
    "paper cup":            ("cup", True),
    "scotch box side":      ("box of Scotch tape", True),
    "bare table":           ("mug", False),   # empty -> should FAIL regardless of noun
}
CURATED = list(CLASS)  # send all of the above

PROMPT = """You are a verification judge for a robot manipulation task.

This is a real top-down photograph of a tabletop. (Two small black dots are
camera-position markers — ignore them.)

TASK THE ROBOT WAS GIVEN: place {noun} on the table.

Decide whether the task SUCCEEDED. Count it a success only if {noun} is actually
present and resting on the table. If none is present, the task FAILED.

Answer on the first line in exactly this format:
VERDICT: PASS
or
VERDICT: FAIL
Then add one short sentence of justification."""

def scene_key(fname):
    b = os.path.splitext(os.path.basename(fname))[0].lower()
    return re.sub(r"\s*dot\s*[12]\s*$", "", b).strip()

def class_of(key):
    for p in sorted(CLASS, key=len, reverse=True):
        if key == p or key.startswith(p + " "):
            return p
    return None

def png_bytes(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    im = im.resize((MAXW, int(round(h * MAXW / w))))
    buf = io.BytesIO(); im.save(buf, "PNG"); return buf.getvalue()

def judge(path, api_key, noun):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)
    prompt = PROMPT.replace("{noun}", "a " + noun)
    contents = [types.Content(role="user", parts=[
        types.Part.from_bytes(data=png_bytes(path), mime_type="image/png"),
        types.Part.from_text(text=prompt),
    ])]
    cfg = types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_level="LOW"))
    text = ""
    for ch in client.models.generate_content_stream(model=MODEL, contents=contents, config=cfg):
        if ch.text: text += ch.text
    m = re.search(r"VERDICT:\s*(PASS|FAIL)", text, re.IGNORECASE)
    v = (m.group(1).upper() == "PASS") if m else (
        True if ("PASS" in text.upper() and "FAIL" not in text.upper()) else
        False if ("FAIL" in text.upper() and "PASS" not in text.upper()) else None)
    return v, text.strip()

def main():
    api_key = None
    for line in open(os.path.join(BASE, ".env")):
        if line.startswith("GEMINI_API_KEY"): api_key = line.split("=", 1)[1].strip()
    assert api_key, "no GEMINI_API_KEY in .env"

    par = {}
    if os.path.exists(PARALLAX_JSON):
        pj = json.load(open(PARALLAX_JSON))
        par = {k: v.get("verdict") for k, v in pj.get("scenes", {}).items()}

    dot1 = {}
    for f in sorted(glob.glob(os.path.join(CUPS, "*.jpeg")) + glob.glob(os.path.join(CUPS, "*.jpg"))):
        if re.search(r"dot\s*1", os.path.basename(f).lower()):
            dot1[scene_key(f)] = f

    # curated subset: first N_PER_CLASS scenes of each curated class
    todo = []
    for cls in CURATED:
        keys = sorted(k for k in dot1 if class_of(k) == cls)[:N_PER_CLASS]
        todo += keys

    print("=" * 100)
    print(f"REAL-PIXEL HEAD-TO-HEAD  ·  Gemini Robotics-ER (RGB, neutral) vs parallax  ·  {FOLDER}  ·  {len(todo)} scenes")
    print("=" * 100)
    print(f"{'scene':<24}{'noun':<20}{'real?':<7}{'Gemini':<8}{'fooled':<8}{'parallax':<10}{'raw'}")
    print("-" * 100)
    out = {}
    n_fool = n_real_pass = n_real = 0
    for key in todo:
        cls = class_of(key); noun, real = CLASS[cls]
        v, text = judge(dot1[key], api_key, noun)
        gem = "PASS" if v else ("FAIL" if v is False else "??")
        fooled = (v is True and not real)          # said "present" when it's a flat print / empty
        pv = par.get(key, "")
        pv_short = "RAISED" if "RAISED" in pv else ("FLAT" if "FLAT" in pv else pv)
        first = text.splitlines()[0] if text else ""
        if real:
            n_real += 1; n_real_pass += int(v is True)
        n_fool += int(fooled)
        print(f"{key:<24}{noun:<20}{str(real):<7}{gem:<8}{('YES' if fooled else '-'):<8}{pv_short:<10}{first[:30]}")
        out[key] = dict(noun=noun, real_object=real, gemini_pass=v, fooled=fooled,
                        parallax=pv, raw=text)
    print("-" * 100)
    print(f"Gemini fooled (PASS on a flat print / empty): {n_fool} / {sum(1 for k in todo if not CLASS[class_of(k)][1])}")
    print(f"Gemini correct on real objects: {n_real_pass} / {n_real}")
    os.makedirs(os.path.dirname(PARALLAX_JSON), exist_ok=True)
    json.dump(out, open(os.path.join(os.path.dirname(PARALLAX_JSON), "realframe_vlm.json"), "w"), indent=2)
    print("raw verdicts saved to", os.path.relpath(os.path.join(os.path.dirname(PARALLAX_JSON), "realframe_vlm.json")))

if __name__ == "__main__":
    main()

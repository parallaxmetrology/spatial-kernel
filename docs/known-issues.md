# Known issues

What the two-view check gets wrong on the real photo set (113 scenes), and why. Counts are from [`results/recorded/cups2/realframe_results.json`](../results/recorded/cups2/realframe_results.json) and reproduce exactly from the photos.

## 1. Transparent objects lose track (3 of 6 clear-glass scenes)

The object is found by its edges and color. A clear glass has little of either, so in half the scenes the tracker latches onto something else between the two views and the shift is implausible (more than 90 px). Those are reported as `detect-fail`, not as a verdict. The other three read correctly as RAISED. Treat glass as outside what this check can see.

## 2. Low, flat-lying real objects can read FLAT (6 of 17)

The check measures height above the table. A Scotch box lying flat (3 of 6), a tape roll (2 of 5 detected) and a box on its side (1 of 6) are real but short, so at this camera distance and 10 cm slide they shift only a little more than the table: −15 px to +1 px, against a 15 px threshold. That's a false *fail*, the safe direction for an auditor, but it's a miss. A longer slide or a closer camera widens the margin.

## 3. Some flat prints read RAISED (4 of 53)

Two cut-out paper cup prints, one flattened box with printed 3D shading and one flat box print read RAISED (15–36 px of relative parallax). Likely causes: paper that curls or lifts off the table, and the tracker picking a different edge in each view. These are the dangerous direction (a decoy passing), so they're worth reproducing with your own prints.

## 4. A print raised on a box is "real" to both cameras

A flat print taped on top of a box *does* stand above the table, so it reads RAISED, correctly by geometry and wrongly for the task. Two camera views can't close this; a third, independent surface can (grasp/proprioception). [`benchmark/THIRD_SURFACE.md`](../benchmark/THIRD_SURFACE.md) shows that in simulation: the proprioception read there is **modeled, not measured**.

## 5. Borderline readings are sensitive to re-encoding

Re-saving the photos as JPEG (quality 90, at the 1,400-px analysis width) moved two scenes by 12–20 px and flipped one verdict (`scotch box 01`, recorded −14.9 px, sits right at the threshold). Run on the original files. The published photo set keeps the original compressed data byte for byte; only metadata was removed.

## 6. One rig, one threshold

The 15 px threshold and 90 px mis-track cap were set on this setup: iPhone 14 Plus, top-down-ish at roughly 40–60 cm, a ~10 cm horizontal slide, analysis at 1,400 px wide. Different distance, baseline or resolution changes the pixel scale. Recalibrate with a known flat print and a known real object before trusting verdicts on a new rig.

## 7. The Gemini comparison is small

27 scenes (9 kinds × 3), one model (`gemini-robotics-er-1.6-preview`), one neutral prompt that names the object, one view per scene, one run. Model answers can change between runs and versions. The recorded answers are verbatim in [`realframe_vlm.json`](../results/recorded/cups2/realframe_vlm.json). It shows that a strong embodied model *can* be fooled by a print. It doesn't estimate how often.

## 8. Older recorded runs

- `synthetic_v1.json` and `synthetic_vlm.json` were recorded with the original six-class suite, before `benchmark/scenes.py` gained the four adversarial classes. Current code reproduces the v2 run exactly, not v1.
- The first placement batch (`results/recorded/placement/`) was recorded shortly before a final edit to the placement script. Rerunning gives gaps within 0.18 cm and the same verdicts. The second batch reproduces exactly.
- In the recorded placement files, the `path` field was rewritten from the author's local folder to `data/...`. Nothing else was changed.

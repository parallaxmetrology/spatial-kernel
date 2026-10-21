# Real-frame capture protocol — iPhone 14 Plus (no LiDAR)

The 14 Plus has no depth sensor, so we get the independent geometric surface from
**two-view parallax**: two photos per scene from a small horizontal offset. The
relative shift of the object vs the table reveals a real raised object from a flat
printed decoy. Pipeline is built and tested (`spatial_kernel/stereo.py`,
`benchmark/test_stereo.py`).

## One-time setup (10 min)

- **Surface:** a plain table. Tape a light outline marking the **bottom-left quadrant**
  (the target region) — roughly the lower-left quarter of what the camera frames.
- **Camera:** normal **Photo** mode (NOT Portrait). Turn on the grid
  (Settings → Camera → Grid). Frame the table roughly top-down-ish at ~40–60 cm.
- **Lock exposure/focus:** long-press to AE/AF-lock so both shots in a pair match.
- **The decoy prop:** print a photo of your real object at **actual size** on paper.
  This flat print, laid in the target region, is the `decoy`.
- **The raised-decoy prop:** the same print taped on top of a small box (~5 cm tall).
- **Calibration:** for the very first scene, also note the object's real height — used
  once to set the parallax threshold.

## The two-view capture (per scene)

1. Frame the scene. Take photo **L**.
2. Slide the phone **~10 cm to the right**, keeping it parallel, same height and angle.
   (Mark two spots on the table edge to make this repeatable.)
3. Take photo **R**. Scene must be static between the two.

That's it — two photos per scene.

## Classes to shoot

| class | what to stage | GT | count |
|---|---|---|---|
| `clean_success` | real object resting in bottom-left | PASS | 8 |
| `wrong_place` | real object in top-right | FAIL | 8 |
| `empty` | nothing on the table | FAIL | 6 |
| **`decoy`** | flat printed photo of the object in bottom-left | FAIL | 8 ← the crux |
| **`raised_decoy`** | printed photo on the ~5 cm box, in bottom-left | FAIL | 6 ← needs 3rd surface |
| `transparent_success` *(optional)* | a clear glass/jar in bottom-left | PASS | 6 |

~42 scenes → ~84 photos, about 30–45 min. A few **matched pairs** (same object & spot,
real vs printed) make the strongest hero comparison — shoot a real `clean_success` and a
`decoy` of the *same* object in the *same* spot back to back.

## Naming & ground truth

Name each file `scene_{class}_{nn}_{L|R}.heic`, e.g. `scene_decoy_03_L.heic`.
HEIC is fine — I convert on ingest. Drop them in `Robotics-ER 1.6/realframes/`.
No separate label file needed; the class is in the filename.

## Transfer

AirDrop or cable the photos into `Robotics-ER 1.6/realframes/`. Tell me when they're
there and I run the real-frame head-to-head: appearance-only, two-view parallax, and
(optionally) the real Gemini VLM, on real pixels — the external-validity result the
synthetic study was built to earn.

## What this closes

- **Realism** — real photographed scenes and a real printed decoy.
- **The hero artifact** — a real photo of a frontier VLM passing a real decoy that
  two-view parallax catches.
- (Stretch) shoot a 1–2 s **video** instead of a still for a few scenes and we also get
  **σ(Ω) temporal** stability — the program's strongest signal — from the frame burst.

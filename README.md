# Spatial Kernel

**Is it really there? A deterministic check of robot task outcomes. No learned judge.**

Ask a vision-language model whether a robot put the mug on the table and it looks at one picture. A picture of a mug is not a mug. On ordinary phone photos of a tabletop, **Gemini Robotics-ER passed 10 of 18 flat printed decoys as real objects present**. A two-view check with standard computer vision caught all 10: take a second photo ~10 cm to the side, and a real object shifts more than the table it stands on, while a print lies in the table plane and shifts with it.

| on the same real photos (27 scenes, 9 kinds) | flat decoys passed as real | real objects passed | empty table passed |
|---|---|---|---|
| Gemini Robotics-ER 1.6, one view, neutral prompt | **10 / 18** | 9 / 9 | 0 / 3 |
| Two-view parallax check (this repo) | **0 / 18** | 9 / 9 | 0 / 3 |

Across the full photo set (113 scenes, 19 object types plus a bare table) the check is right on 99 of 109 cleanly detected scenes and loses track of the object in 4. Where it fails is in [docs/known-issues.md](docs/known-issues.md), and it's worth reading: the model and the geometry fail in *different* places, which is the point.

**Play it:** [Real or printed?](https://parallaxmetrology.github.io/spatial-kernel/) shows you what the model saw. Call it, then see the second view.

## Install

```bash
pip install "spatial-kernel @ git+https://github.com/parallaxmetrology/spatial-kernel"
```

## Use

Two photos of a scene from positions ~10 cm apart, same height and angle, with two small black dots on the table (the plane's reference). The [capture protocol](benchmark/CAPTURE_PROTOCOL.md) takes ten minutes to set up.

```bash
spatial-kernel twoview left.jpg right.jpg        # RAISED, FLAT, no-object or detect-fail, with the pixel shifts
spatial-kernel verify scene.png --task bottom_left --json-out report.json --overlay-out overlay.png
spatial-kernel verify scene.png --task bottom_left --depth depth.npy --table-z 0.0 --isolate depth_plane --parallax
```

```python
from spatial_kernel.twoview import check_pair
from spatial_kernel import verify_spatial_task, verify_parallax, fuse_reads, Read
```

- **`twoview`**: the real-photo object/print test.
- **`verify`**: did the object land in the task region? Region occupancy and leakage, a correction vector, optional depth, object isolation (`--isolate`), and `--parallax`, which makes success the agreement of independent reads (RGB, depth, proprioception, time) with their disagreement Ω as the confidence. Reads from one surface can't certify anything on their own: a single-surface verdict is capped at 0.5 confidence.
- **Gemini is optional** and stays downstream: `--robotics-er` asks it to explain a report the deterministic layer already decided. Set `GEMINI_API_KEY`; nothing is sent otherwise.

## The design idea

Learned eyes, deterministic judge. A segmenter or a VLM may say *which pixels* are the mug; whether the task succeeded is decided by geometry that can't be talked into a pass. When the independent reads disagree, the outcome is **contested**, not passed.

## Evidence, and how to reproduce it

| result | recorded in | reproduce |
|---|---|---|
| Real photos, two-view verdicts (113 scenes) | [`results/recorded/cups2/realframe_results.json`](results/recorded/cups2/realframe_results.json) | `python tools/fetch_photos.py && python benchmark/realframe_parallax.py` (exact) |
| Gemini on the same photos (27 scenes, verbatim answers) | [`results/recorded/cups2/realframe_vlm.json`](results/recorded/cups2/realframe_vlm.json) | `python -m benchmark.realframe_vlm` (needs a key; model answers can vary) |
| Synthetic, pre-registered, 1,200 scenes, 10 classes | [`results/recorded/synthetic_v2.json`](results/recorded/synthetic_v2.json), [RESULTS_v2.md](benchmark/RESULTS_v2.md) | `python -m benchmark.run_headtohead --per-class 120` (exact) |
| Placement precision, real objects vs a target (cm) | [`results/recorded/placement*/`](results/recorded/) | `python -m benchmark.placement_precision placement2` |

The synthetic v2 run is the honest one: its predictions were filed before it ran ([PREREGISTRATION.md](benchmark/PREREGISTRATION.md)), and it adds classes that *break* depth. Overall accuracy there favors the RGB-only judge (80% vs 70%), because transparent glass and depth dropout defeat the depth read. Per failure mode: parallax removes every flat-decoy false pass (120 → 0), and a print mounted on a box fools both surfaces. More in [docs/evidence.md](docs/evidence.md).

## Not in this repository

The research also includes an **authority-separated auditor** for robot outcomes and VLA training data: a system-owned contract decides what counts as success, independent evidence executes it, and missing evidence yields ABSTAIN instead of an inferred PASS. Its study design is [public](https://www.parallaxmetrology.com/spatial_pm.html); the code isn't here. The *Confidence without correctness* result (a learned box moved 2.4% while the object's true visible extent moved 67.7%) comes from that work and isn't reproduced by this repo.

If you run robot evaluation or a VLA data pipeline and want to test it on your data: **russellgparrish@gmail.com**.

## Contribute

- Shoot your own pairs (phone, two dots, ~10 cm slide) and post what reads wrong. Especially objects it misses: low boxes, glass, curled paper.
- Try other VLMs on [the same photos](tools/fetch_photos.py) with the same prompt.
- Break it: a decoy that reads RAISED, or a real object that reads FLAT.

See [CONTRIBUTING.md](CONTRIBUTING.md). Discussion: [Discussions](https://github.com/parallaxmetrology/spatial-kernel/discussions).

## Part of Parallax Metrology

One principle across domains: structure has geometry, and geometry is measurable without knowing what it means. Siblings: [Visual Thinking Lens](https://github.com/parallaxmetrology/visual-thinking-lens) (composition), [Linguistic Kernel](https://github.com/parallaxmetrology/linguistic-kernel) (text), [Inspection Coordinates](https://github.com/parallaxmetrology/inspection-coordinates) (semiconductor inspection). Background: [the Spatial Kernel paper](https://www.parallaxmetrology.com/SPATIAL_KERNEL_PAPER.html). Lineage: [docs/lineage.md](docs/lineage.md).

## License and data

Code: [MIT](LICENSE). Docs and photos: [CC BY 4.0](LICENSE-CONTENT). The photos are the author's own, taken on an iPhone 14 Plus, with all metadata removed ([`tools/strip_metadata.py`](tools/strip_metadata.py), lossless).

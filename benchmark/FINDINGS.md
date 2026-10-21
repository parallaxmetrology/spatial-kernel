# Head-to-head: deterministic parallax vs an appearance-only judge

**Question (as posed):** run the multi-observable parallax success-detector against a
judge that only looks at the picture, on a labeled suite — report where they disagree
and who was right.

**Result (24 scenes, 6 classes, ground-truth labeled, seed 7, real VLM run):**

| judge | overall accuracy | false PASSES | decoy class |
|---|---|---|---|
| appearance_only (RGB proxy, geometry-blind) | 83.3% | 4 | 0% |
| **real VLM — Gemini Robotics-ER, RGB-only** | **83.3%** | **4** | **0%** |
| **parallax (RGB + depth agreement, Ω)** | **100%** | **0** | **100%** |

The real frontier embodied-reasoning VLM lands on the **exact same score and the exact
same error** as the geometry-blind proxy: it passes every decoy. That dual result is
the point — it confirms (a) the proxy was a faithful stand-in, and (b) the blind spot
is not a weakness of *one* model but of the *single-RGB-frame setup itself*.

**Where they diverge — and who is right:** exactly one class, `decoy`.

```
[decoy] gt=FAIL  appearance=PASS  vlm=PASS  parallax=FAIL  →  correct: PARALLAX  (4/4)
```

Gemini's own words on a decoy: *"VERDICT: PASS — The dark block is located entirely
within the bottom-left quadrant."* It is reasoning correctly about the pixels; the
pixels just do not contain the fact that no object was placed.

A `decoy` is a flat printed photo of the object sitting in the target region. In RGB
it is indistinguishable from a correct placement — a perfect segmenter masks it, the
centroid is in-region, so the appearance judge passes it with high confidence. It is
*wrong*: nothing was placed. The depth surface sees no object above the table plane,
the two reads disagree (Ω spikes), and parallax correctly returns **contested → FAIL**.

Every other class (clean success, wrong place, clutter, occlusion, empty) both judges
get right. The benchmark isolates a single mechanism: **appearance does not carry
geometry.** Where the discriminator between PASS and FAIL is geometric, an RGB-only
judge — deterministic *or* a VLM looking at one frame — must guess; an independent
geometric surface resolves it.

## Why the "false PASSES" row is the real headline

For a verifier's two roles — a **training-data filter** for a VLA data engine, and a
**safety gate** before a robot acts — the dangerous error is the *false pass*: certifying
a failed outcome as success. It poisons the training set and green-lights bad actions.
On this suite both the appearance-only judge and the real VLM produce 4 false passes;
parallax produces 0. That is the number that matters — and the VLM-as-judge is
*empirically*, not just hypothetically, exposed to it on the decoy class.

## Honest limitations (do not oversell)

- **Synthetic, schematic stimuli.** These are generated top-down scenes, not real
  rollouts, and the VLM was told so. The decoy is RGB-*identical* to a real placement
  by construction (same draw call; only depth differs). This makes the test a clean
  probe of an **information limit** — a single RGB frame does not contain depth — not a
  measure of how often decoys occur in the wild, nor of whether a VLM might catch a
  *photorealistic* printed decoy via shadow/texture cues a cartoon lacks. The claim is
  the information-theoretic one, and the real VLM confirmed it.
- **The VLM engaged fairly.** Neutral judge prompt, RGB-only (the same frame the
  appearance judge sees; only parallax also gets depth). Gemini did not refuse — it gave
  confident, coherent, *wrong* verdicts on the decoy. No answer was fabricated; the raw
  text is in `results/recorded/synthetic_vlm.json`.
- **Parallax's win depends on a usable depth surface.** Its advantage is the independent
  observable, not the kernel; remove depth and it degrades to the RGB read.

## Reproduce the VLM run

The real-VLM slot is wired (`benchmark.vlm_judge.vlm_judge_rgb`, RGB-only judge prompt).
Put `GEMINI_API_KEY=...` in `.env` (gitignored) and:

```bash
python3 -m benchmark.run_headtohead --vlm --per-class 4 --json-out results/recorded/synthetic_vlm.json
```

Result on this run: real Gemini Robotics-ER = 83.3% overall, 4 false passes, 0% on
decoy — matching the geometry-blind proxy exactly.

## Reproduce

```bash
python3 -m benchmark.run_headtohead --json-out results/recorded/synthetic_v1.json
```

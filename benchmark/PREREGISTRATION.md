# Pre-registration — adversarial-symmetric benchmark v2

**Written before running v2.** The point of this file is to commit to predictions —
especially predictions of where OUR method fails — before seeing the numbers. If the
results contradict these, that is a finding, not something to quietly retune.

## What changed from v1

v1 only contained classes that break RGB-only judges. That is the cherry-pick we are
correcting. v2 adds classes designed to break **parallax** (the depth surface), so the
benchmark can show where our own method loses.

## Classes and predicted verdicts

GT = ground truth. ✓ = predicted correct, ✗ = predicted wrong.

| class | GT | appearance (RGB) | VLM (RGB) | parallax (RGB+depth) |
|---|---|---|---|---|
| clean_success | PASS | ✓ PASS | ✓ PASS | ✓ PASS |
| wrong_place | FAIL | ✓ FAIL | ✓ FAIL | ✓ FAIL |
| clutter_success | PASS | ✓ PASS | ✓ PASS | ✓ PASS |
| occluded_success | PASS | ✓ PASS | ✓ PASS | ✓ PASS |
| empty | FAIL | ✓ FAIL | ✓ FAIL | ✓ FAIL |
| **decoy** (flat print) | FAIL | ✗ PASS | ✗ PASS | ✓ FAIL |
| **transparent_success** (glass; depth sees through) | PASS | ✓ PASS | ✓ PASS | **✗ FAIL** |
| **depth_dropout_success** (object occluded from depth cam) | PASS | ✓ PASS | ✓ PASS | **✗ FAIL** |
| **depth_noise_success** (noisy depth) | PASS | ✓ PASS | ✓ PASS | **~ mostly PASS, some ✗** |
| **raised_decoy** (printed photo on a box) | FAIL | ✗ PASS | ✗ PASS | **✗ PASS** |

## Pre-stated predictions (commit these)

1. **Parallax will NOT be 100% on v2.** It will false-FAIL real successes where depth
   lies (transparent, dropout) and false-PASS the raised_decoy where depth and RGB are
   co-fooled. Expected parallax accuracy: **70–85%**, down from v1's 100%.
2. **The flat-decoy advantage holds.** On `decoy`, RGB-only judges (incl. VLM) still
   false-pass; parallax still catches it. That result is real and survives.
3. **raised_decoy is the key humility case.** Two surfaces (RGB + depth) that agree can
   still be jointly fooled by a 3D decoy. Predicted: parallax PASSES it (wrong). This is
   the evidence that 2 surfaces is a floor, not a ceiling — robustness needs a *third*
   independent surface (proprioception/grasp), exactly the N-independence thesis.
4. **No judge will be best everywhere.** RGB-only wins on transparent/dropout (where
   depth fails); parallax wins on flat decoy; nobody catches raised_decoy without a
   third surface. The honest conclusion is not "parallax wins" but "robustness scales
   with the number of genuinely independent surfaces, and any fixed set has an
   adversarial complement."

## Statistics plan (pre-committed)

- per_class = 120, seeds vary scene jitter (position, noise, clutter). Total ~1200.
- Report per-class accuracy and false-pass rate with **bootstrap 95% CIs** (10k resamples).
- VLM run on a stratified subset only (cost), reported separately and labeled as subset.
- Headline metric for the safety framing: **false-pass count/rate per judge.**

## What would falsify the thesis

- If parallax stays ~100% on v2, the adversarial classes are too weak — rebuild them.
- If RGB-only judges also catch the flat decoy at scale, the v1 result was a fluke.
- If a third surface does NOT rescue raised_decoy in a later test, the N-independence
  claim is wrong.

*Filed before the v2 run. Russell Parrish / VTL spatial.*

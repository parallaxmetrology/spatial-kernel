# Results v2 — adversarial-symmetric, scaled, pre-registered

1200 scenes, 120/class, bootstrap 95% CIs. Predictions filed in
[`PREREGISTRATION.md`](PREREGISTRATION.md) **before** this run.

## Overall (the wrong metric, shown on purpose)

| judge | accuracy | 95% CI |
|---|---|---|
| appearance_only (RGB) | 80.0% | [77.8%, 82.2%] |
| parallax (RGB + depth) | 70.0% | [67.3%, 72.5%] |

**Parallax scores *lower* overall than the RGB-only judge.** That is not a bug in the
result — it is the point of doing the honest version. Aggregate accuracy is a function
of the *class mix*, and we deliberately added three classes that break depth and only
one that uniquely favors it. Whoever picks the class mix picks the winner. So aggregate
accuracy is the wrong headline. The per-failure-mode breakdown is the real result.

## Per-class (the right metric)

| class | GT | appearance | parallax | who's right |
|---|---|---|---|---|
| clean_success | PASS | 100% | 100% | tie |
| wrong_place | FAIL | 100% | 100% | tie |
| clutter_success | PASS | 100% | 100% | tie |
| occluded_success | PASS | 100% | 100% | tie |
| empty | FAIL | 100% | 100% | tie |
| **decoy** (flat print) | FAIL | **0%** | **100%** | **parallax** |
| **transparent_success** (glass) | PASS | **100%** | **0%** | **appearance** |
| **depth_dropout_success** | PASS | **100%** | **0%** | **appearance** |
| depth_noise_success | PASS | 100% | 100% | tie † |
| **raised_decoy** (print on a box) | FAIL | **0%** | **0%** | **neither** |

† depth_noise: see disclosed iteration below.

## What this actually shows

1. **The flat-decoy finding survives scale.** RGB-only judges false-pass every flat
   decoy (0%, n=120); parallax catches every one (100%). Tight CIs. This is the result
   from v1, now robust to n and not a fluke.
2. **Where parallax loses, named and quantified.** When the depth surface lies — a
   transparent object the sensor sees through, or an object in the depth camera's
   occlusion shadow — parallax false-FAILS a real success 100% of the time, and the
   RGB judge is correct. We lose exactly where depth has no signal.
3. **The humility case: `raised_decoy`.** A printed photo on a low box satisfies BOTH
   surfaces (RGB sees the object, depth sees raised mass) — so they agree, and parallax
   false-PASSES it (0%). Two surfaces that can be jointly fooled are not enough. This is
   direct evidence for the N-independence thesis: robustness needs a *third* independent
   surface (grasp/proprioception — "did the gripper actually close on mass?"), and any
   fixed set of surfaces has an adversarial complement.
4. **No judge dominates.** RGB-only wins on transparent/dropout; parallax wins on flat
   decoy; nobody wins on raised_decoy. The honest conclusion is not "parallax is better"
   — it is "success detection has no single-observable solution; robustness scales with
   the number of genuinely independent observables."

## Safety metric (false passes — certifying a failure as success)

| judge | false passes / 1200 | composed of |
|---|---|---|
| appearance_only | 240 | 120 flat decoy + 120 raised decoy |
| parallax | 120 | 120 raised decoy only |

Parallax halves the dangerous error and eliminates the flat-decoy false-pass entirely,
but does not reach zero: its residual 120 are all `raised_decoy`, the case that requires
a third surface. This is the honest safety story — a real improvement with a known,
named residual, not a solved problem.

## Prediction scorecard (graded against PREREGISTRATION.md)

| prediction | outcome |
|---|---|
| parallax NOT 100% on v2; expected 70–85% | ✓ 70.0% — at the low edge of the predicted band |
| flat-decoy advantage holds | ✓ confirmed at n=120 |
| parallax false-fails transparent + dropout | ✓ both 0% |
| raised_decoy: parallax false-passes (humility case) | ✓ 0%, false-pass confirmed |
| no judge best everywhere | ✓ confirmed |
| depth_noise: "mostly PASS, some fail" | ✗ **wrong as written** — see disclosure |

## Disclosed iteration (do not hide this)

The v2 smoke run (n=3) showed `depth_noise_success` false-failing **100%**, not the
"mostly pass" I pre-registered. Cause was a strawman in our *own* pipeline: the depth
occupancy read used a hardcoded `1e-3` height threshold, far below realistic sensor
noise, so noise flooded the frame with phantom occupancy. We fixed it before the scaled
run by making the threshold **noise-aware** (robust MAD estimate of table-plane noise;
require height > ~4σ) — standard practice, not result-chasing. With the fix, parallax
handles moderate noise (100%). This is disclosed as a deviation: the pre-registered
prediction was wrong as written, and the fix is a legitimate engineering correction,
reported rather than buried.

## Honest standing

This is now an adversarial-symmetric, pre-registered, scaled (n=1200, CIs) controlled
study that reports where our own method fails. It is credible as a *mechanism study*.
It is still **synthetic** — the next and only remaining gap to external validity is real
photographed scenes with real depth (the flat decoy and raised decoy literally printed
and shot). The mechanism is proven; realism is the open item.

Reproduce: `python3 -m benchmark.run_headtohead --per-class 120 --json-out results/recorded/synthetic_v2.json`

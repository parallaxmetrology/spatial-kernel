"""Head-to-head: parallax vs appearance-only (vs optional real VLM) on labeled scenes.

Reports overall accuracy, per-class accuracy, and — the question that matters — the
scenes where the judges DISAGREE and which one matched ground truth.

Run:  python3 -m benchmark.run_headtohead
      python3 -m benchmark.run_headtohead --vlm --gemini-api-key KEY   # real VLM
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict

from .judges import judge_appearance_only, judge_parallax, judge_vlm
from .scenes import CLASSES, generate


def _bootstrap_ci(bits: list[int], n_resamples: int = 10000, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap CI for the mean of a 0/1 list."""
    if not bits:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(bits)
    means = []
    for _ in range(n_resamples):
        s = sum(bits[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo = means[int(0.025 * n_resamples)]
    hi = means[int(0.975 * n_resamples)]
    return (lo, hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--vlm", action="store_true", help="also run a real Gemini Robotics-ER judge")
    ap.add_argument("--gemini-api-key", default=os.environ.get("GEMINI_API_KEY"))
    ap.add_argument("--env-file", default=".env", help="load KEY=VALUE pairs (e.g. GEMINI_API_KEY)")
    ap.add_argument("--json-out")
    args = ap.parse_args()

    # Load .env so the key never has to be passed on the command line.
    if args.vlm and not args.gemini_api_key and os.path.exists(args.env_file):
        for line in open(args.env_file):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        args.gemini_api_key = os.environ.get("GEMINI_API_KEY")

    scenes = generate(per_class=args.per_class, seed=args.seed)
    api_key = args.gemini_api_key if args.vlm else None

    rows = []
    vlm_ran = False
    for sc in scenes:
        a_pass, a_conf, a_note = judge_appearance_only(sc)
        p_pass, p_conf, p_note = judge_parallax(sc)
        v = judge_vlm(sc, api_key)
        if v is not None:
            vlm_ran = True
        rows.append({
            "cls": sc.cls, "gt": sc.gt_pass,
            "appearance": {"pass": a_pass, "conf": a_conf, "note": a_note, "correct": a_pass == sc.gt_pass},
            "parallax": {"pass": p_pass, "conf": p_conf, "note": p_note, "correct": p_pass == sc.gt_pass},
            "vlm": None if v is None else {"pass": v[0], "conf": v[1], "note": v[2], "correct": v[0] == sc.gt_pass},
        })

    n = len(rows)
    acc = lambda key: sum(r[key]["correct"] for r in rows) / n
    ci = lambda key: _bootstrap_ci([1 if r[key]["correct"] else 0 for r in rows])
    print("=" * 74)
    print(f"HEAD-TO-HEAD v2 (adversarial-symmetric)  ·  {n} scenes  ·  {args.per_class}/class")
    print("=" * 74)
    a_lo, a_hi = ci("appearance"); p_lo, p_hi = ci("parallax")
    print("\nOVERALL ACCURACY (95% bootstrap CI)")
    print(f"  appearance_only : {acc('appearance'):.1%}  [{a_lo:.1%}, {a_hi:.1%}]")
    print(f"  parallax        : {acc('parallax'):.1%}  [{p_lo:.1%}, {p_hi:.1%}]")
    if vlm_ran:
        v_lo, v_hi = ci("vlm")
        print(f"  vlm (subset)    : {sum(r['vlm']['correct'] for r in rows)/n:.1%}  [{v_lo:.1%}, {v_hi:.1%}]")

    print("\nPER-CLASS ACCURACY")
    vh = f"  {'vlm':>7}" if vlm_ran else ""
    print(f"  {'class':<18}{'n':>3}  {'appearance':>11}  {'parallax':>9}{vh}")
    by_cls = defaultdict(list)
    for r in rows:
        by_cls[r["cls"]].append(r)
    for cls in CLASSES:
        rs = by_cls[cls]
        a = sum(x["appearance"]["correct"] for x in rs) / len(rs)
        p = sum(x["parallax"]["correct"] for x in rs) / len(rs)
        vc = ""
        if vlm_ran:
            v = sum(x["vlm"]["correct"] for x in rs) / len(rs)
            vc = f"  {v:>6.0%}"
        flag = "  <-- divergence" if abs(a - p) > 1e-9 else ""
        print(f"  {cls:<18}{len(rs):>3}  {a:>10.0%}  {p:>8.0%}{vc}{flag}")

    # The question: where do they disagree, and who is right?
    disagree = [r for r in rows if r["appearance"]["pass"] != r["parallax"]["pass"]]
    print(f"\nDISAGREEMENTS (appearance vs parallax): {len(disagree)} / {n}")
    seen = set()
    for r in disagree:
        if r["cls"] in seen:
            continue
        seen.add(r["cls"])
        winner = "parallax" if r["parallax"]["correct"] else ("appearance" if r["appearance"]["correct"] else "neither")
        print(f"  [{r['cls']:<16}] gt={'PASS' if r['gt'] else 'FAIL'}  "
              f"appearance={'PASS' if r['appearance']['pass'] else 'FAIL'}  "
              f"parallax={'PASS' if r['parallax']['pass'] else 'FAIL'}  →  correct: {winner.upper()}")

    # Safety-relevant subset: false PASSES (judge passed a real failure).
    fp = lambda key: sum(1 for r in rows if r[key] is not None and r[key]["pass"] and not r["gt"])
    vlm_fp = f"    vlm: {fp('vlm')}" if vlm_ran else ""
    print(f"\nFALSE PASSES (rubber-stamped a failed outcome)   "
          f"appearance_only: {fp('appearance')}    parallax: {fp('parallax')}{vlm_fp}")
    print("  ^ this is the dangerous error for a training-data filter or a safety gate.")
    print("=" * 74)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"n": n, "accuracy": {"appearance": acc("appearance"), "parallax": acc("parallax")},
                       "rows": rows}, f, indent=2)
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

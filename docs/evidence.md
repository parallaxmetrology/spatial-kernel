# Evidence

## Real photos: two-view parallax (exact reproduction)

113 scenes, each two photos ~10 cm apart: 19 object types (real mugs, cups, glass, tape, boxes; flat prints and cut-outs of the same objects; prints raised on a box) plus a bare table. Ground truth comes from the file name.

```bash
python tools/fetch_photos.py
python benchmark/realframe_parallax.py          # writes results/cups2/, compare with results/recorded/cups2/
```

- Cleanly detected and graded: 109 scenes, **99 correct**. Detection failures: 4 (three clear glass, one tape roll).
- Relative parallax at 1,400 px wide (10th–90th percentile): flat prints 2–13 px (median 6), real objects standing on the table 13–70 px (median 38). The ranges touch at the threshold, which is where the misses are.
- The failures are listed in [known-issues.md](known-issues.md).

## Real photos: Gemini Robotics-ER vs the check

27 of those scenes (3 each of 9 kinds), first view only, a neutral prompt that names the object and says nothing about decoys (the prompt is in [`benchmark/realframe_vlm.py`](../benchmark/realframe_vlm.py)).

| kind | real? | Gemini PASS | two-view verdict correct |
|---|---|---|---|
| full-page print of a paper cup | no | 3 / 3 | 3 / 3 |
| flattened box with printed 3D shading | no | 3 / 3 | 3 / 3 |
| flat print of a Scotch box | no | 3 / 3 | 3 / 3 |
| full-page print of a mug | no | 1 / 3 | 3 / 3 |
| cut-out print of a mug | no | 0 / 3 | 3 / 3 |
| real mug, paper cup, box on its side | yes | 9 / 9 | 9 / 9 |
| bare table | — | 0 / 3 | 3 / 3 |

Gemini passed 10 of 18 decoys and the check caught all 10. Gemini wasn't careless: on the cut-outs it said things like "appears to be a flat sticker or image of a mug". Where the print is good, one view doesn't hold the answer.

## Placement precision (real objects, cm)

Single photos with a green target and a real object at some offset. The check measures the gap in cm from the fiducial dots (2 cm tolerance). Gemini is asked for PASS/FAIL and to point at object and target.

- 22 measured scenes over two batches. Gemini's verdict agreed with the measurement in all 22; no false passes here.
- What the measurement adds is the number: the gap and the correction vector, in cm. Gemini's own points implied gaps usually within 1 cm of the measured ones, at most 2.2 cm off.

So, honestly: with real objects placed visibly off target, the model's judgment was fine. Its failure on this rig was the decoys.

## Synthetic, pre-registered (exact reproduction)

`python -m benchmark.run_headtohead --per-class 120` reproduces [`synthetic_v2.json`](../results/recorded/synthetic_v2.json): 1,200 rendered scenes, 10 classes, predictions filed first ([PREREGISTRATION.md](../benchmark/PREREGISTRATION.md)), one disclosed fix ([RESULTS_v2.md](../benchmark/RESULTS_v2.md)). Classes include ones that break depth (glass, depth dropout) and one that breaks both cameras (a print on a box). Read the per-class table, not the overall accuracy: the class mix decides the overall winner. The synthetic separation is cleaner than any real sensor; the real photos are the load-bearing evidence.

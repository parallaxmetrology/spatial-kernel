# Contributing

Most useful contributions are photos and robot data that the check gets wrong.

- **Your own pairs.** Two black dots on a table, two photos ~10 cm apart ([capture protocol](benchmark/CAPTURE_PROTOCOL.md)), then `spatial-kernel twoview left.jpg right.jpg`. Report what misreads, with the `relative_parallax_px` values. Glass, low objects, curled paper and shiny surfaces are all interesting.
- **Other models.** Run another VLM on the published photos with the prompt in `benchmark/realframe_vlm.py` and share its answers.
- **Robot data.** Logged episodes with two camera views or depth: does the parallax read agree with what happened?

## Development

```bash
git clone https://github.com/parallaxmetrology/spatial-kernel
cd spatial-kernel
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                          # photo-based tests skip until: python tools/fetch_photos.py
```

## The firm rule

Recorded results in `results/recorded/` are never edited to match new code. If a change moves the real-photo verdicts, it gets its own version, a changelog entry and a new recorded run beside the old one. `tests/test_twoview.py` must keep matching the recorded scenes.

Be kind and specific. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

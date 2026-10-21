"""Two-view check on drawn scenes: a raised object shifts more than the table plane
between the views; a flat print shifts with it. Uses the published threshold."""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from spatial_kernel.twoview import check_pair

ROOT = Path(__file__).resolve().parents[1]


def scene(path, table_dx, object_dx, with_object=True):
    im = Image.new("RGB", (1400, 1050), (214, 214, 210))
    d = ImageDraw.Draw(im)
    for x in (560, 820):  # fiducial dots, lower region
        d.ellipse([x + table_dx - 22, 820, x + table_dx + 22, 864], fill=(12, 12, 12))
    if with_object:
        x0 = 380 + object_dx
        d.rectangle([x0, 300, x0 + 180, 480], fill=(190, 40, 40))
    im.save(path)
    return path


@pytest.mark.parametrize("object_dx,expected", [(-60, "FLAT"), (-110, "RAISED")])
def test_flat_vs_raised(tmp_path, object_dx, expected):
    a = scene(tmp_path / "a.png", 0, 0)
    b = scene(tmp_path / "b.png", -60, object_dx)
    res = check_pair(str(a), str(b))
    assert res["table_shift_px"] == pytest.approx(-60, abs=1)
    assert res["verdict"] == expected


def test_empty_table(tmp_path):
    a = scene(tmp_path / "a.png", 0, 0, with_object=False)
    b = scene(tmp_path / "b.png", -60, 0, with_object=False)
    assert check_pair(str(a), str(b))["verdict"] == "no-object"


RECORDED = ROOT / "results" / "recorded" / "cups2" / "realframe_results.json"
PHOTOS = ROOT / "data" / "cups2"


@pytest.mark.skipif(not PHOTOS.is_dir(), reason="photos not downloaded (python tools/fetch_photos.py)")
@pytest.mark.parametrize("scene_name", ["full page paper cup 01", "real mug 01", "scotch box flat 01",
                                        "cut out mug on the box 01", "bare table 01"])
def test_matches_recorded(scene_name):
    rec = json.loads(RECORDED.read_text())["scenes"][scene_name]
    files = {p.name.lower(): p for p in PHOTOS.iterdir()}
    v1, v2 = (str(files[f"{scene_name} dot{i}.jpeg"]) for i in (1, 2))
    res = check_pair(v1, v2)
    assert res.get("relative_parallax_px") == rec["relative_parallax_px"]
    assert np.isclose(res["table_shift_px"], rec["table_shift_px"])

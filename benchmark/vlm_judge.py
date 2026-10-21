"""Real Gemini Robotics-ER as a VLM-as-judge — fair single-RGB-frame setup.

The VLM receives ONLY the RGB image and the task. It is NOT given the deterministic
metrics or the depth map (that would leak the answer). This is the honest comparison:
every judge that sees only one RGB frame — deterministic proxy or frontier VLM —
shares the same blind spot where geometry is the discriminator.

The verdict is parsed from a forced single line ("VERDICT: PASS|FAIL").
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import cv2
import numpy as np

DEFAULT_MODEL = "gemini-robotics-er-1.6-preview"

JUDGE_PROMPT = """You are a verification judge for a robot manipulation task.

This is a top-down, schematic rendering of a tabletop workspace. A dark rectangular
block may be present somewhere on the table.

TASK THE ROBOT WAS GIVEN: place the block in the BOTTOM-LEFT quadrant of the workspace
(the lower-left quarter of the image).

Decide whether the task SUCCEEDED. Count it a success only if a block is actually
present and resting in the bottom-left quadrant. If no block is present, or it is in
the wrong quadrant, the task FAILED.

Answer on the first line in exactly this format:
VERDICT: PASS
or
VERDICT: FAIL
Then add one short sentence of justification."""


def _parse(text: str) -> bool | None:
    m = re.search(r"VERDICT:\s*(PASS|FAIL)", text, re.IGNORECASE)
    if m:
        return m.group(1).upper() == "PASS"
    up = text.upper()
    if "PASS" in up and "FAIL" not in up:
        return True
    if "FAIL" in up and "PASS" not in up:
        return False
    return None


def vlm_judge_rgb(rgb: np.ndarray, api_key: str, model: str = DEFAULT_MODEL) -> tuple[bool | None, str]:
    """Return (pass_bool_or_None, raw_text). None = unparseable/refused."""
    from google import genai
    from google.genai import types

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "frame.png"
        cv2.imwrite(str(p), rgb)
        image_bytes = p.read_bytes()

    client = genai.Client(api_key=api_key)
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                types.Part.from_text(text=JUDGE_PROMPT),
            ],
        )
    ]
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level="LOW")
    )
    text = ""
    for chunk in client.models.generate_content_stream(model=model, contents=contents, config=config):
        if chunk.text:
            text += chunk.text
    return _parse(text), text.strip()

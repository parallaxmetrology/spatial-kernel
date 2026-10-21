from __future__ import annotations

from typing import Any


DEFAULT_MODEL = "gemini-robotics-er-1.6-preview"


def build_robotics_er_prompt(report: dict[str, Any]) -> str:
    task = report["task"]
    compliance = report["compliance"]
    metrics = report["visual_field"]
    region = report["region"]
    depth = report.get("depth_field")

    target_lines = "\n".join(
        f"  {k}: measured={float(metrics.get(k, 0.0)):+.4f} "
        f"target={float(v):+.4f} gap={float(compliance['gaps'].get(k, 0.0)):+.4f}"
        for k, v in task["target"].items()
    )
    depth_block = ""
    if depth:
        depth_block = "\nDEPTH FIELD:\n" + "\n".join(f"  {k}: {v}" for k, v in depth.items())

    return f"""You are reviewing a deterministic spatial task-compliance report.

Do not decide pass/fail yourself. The deterministic verifier already produced the status.
Use the image only to explain the geometric failure or success in field terms.

TASK:
  name: {task['name']}
  target_region: {task['target_region']}
  notes: {task.get('notes') or ''}

COMPLIANCE:
  status: {compliance['status']}
  score: {float(compliance['score']):.4f}
  threshold: {float(task['threshold']):.4f}
  largest_gap: {compliance['largest_gap']}
  failure_modes: {', '.join(compliance['failure_modes'])}
  correction: {compliance['correction']}

KERNEL MEASUREMENTS VS TARGET:
{target_lines}

FULL VISUAL FIELD:
  delta_x={float(metrics['delta_x']):+.4f}
  delta_y={float(metrics['delta_y']):+.4f}
  r_v={float(metrics['r_v']):.4f}
  rho_r={float(metrics['rho_r']):.4f}
  mu={float(metrics['mu']):.6f}
  x_p={float(metrics['x_p']):.4f}
  theta={float(metrics['theta']):.4f}
  d_s={float(metrics['d_s']):.6f}
  sdi={float(metrics['sdi']):.4f}

REGION OCCUPANCY:
  target_mass_ratio={float(region['target_mass_ratio']):.4f}
  leakage_ratio={float(region['leakage_ratio']):.4f}
  target_region_density={float(region['target_region_density']):.4f}
  target_region_void={float(region['target_region_void']):.4f}
{depth_block}

Respond with four short sections:
1. Geometric compliance summary
2. Failure mechanism or success mechanism
3. Correction implication
4. Perception-vs-field disagreement, if any

Avoid object names unless needed to locate the field mechanism. Prefer mass, void,
centroid, leakage, density, dispersion, region occupancy, and correction language."""


def analyze_with_robotics_er(
    image_path: str,
    report: dict[str, Any],
    api_key: str,
    model: str = DEFAULT_MODEL,
) -> str:
    try:
        from google import genai
        from google.genai import types
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("google-genai is required for Robotics-ER analysis") from exc

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    ext = image_path.lower().rsplit(".", 1)[-1]
    mime = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
    }.get(ext, "image/png")

    client = genai.Client(api_key=api_key)
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime),
                types.Part.from_text(text=build_robotics_er_prompt(report)),
            ],
        )
    ]
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level="HIGH")
    )
    response = ""
    for chunk in client.models.generate_content_stream(model=model, contents=contents, config=config):
        if chunk.text:
            response += chunk.text
    return response

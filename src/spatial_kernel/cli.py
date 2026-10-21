from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from .core import KernelConfig
from .robotics_er import analyze_with_robotics_er, build_robotics_er_prompt
from .verify import report_to_json, verify_parallax, verify_spatial_task
from .visualize import write_overlay


def parse_roi(value: str | None) -> tuple[int, int, int, int] | None:
    if not value:
        return None
    parts = [int(p.strip()) for p in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI must be x,y,w,h")
    return tuple(parts)  # type: ignore[return-value]


def cmd_verify(args: argparse.Namespace) -> int:
    image_path = Path(args.image).expanduser()
    if not image_path.exists():
        print(
            f"Image not found: {args.image}\n"
            "Replace the placeholder with a real image path, for example:\n"
            '  python3 -m spatial_kernel.cli verify path/to/image.png --task bottom_left',
            file=sys.stderr,
        )
        return 1

    if args.depth:
        depth_path = Path(args.depth).expanduser()
        if not depth_path.exists():
            print(f"Depth map not found: {args.depth}", file=sys.stderr)
            return 1
    else:
        depth_path = None

    config = KernelConfig(target_max_side=args.max_side)
    if args.parallax:
        report = verify_parallax(
            image_path=str(image_path),
            task=args.task,
            depth_path=str(depth_path) if depth_path else None,
            table_z=args.table_z,
            roi=args.roi,
            config=config,
            rgb_isolate=args.isolate if args.isolate != "none" else "largest_component",
        )
        payload = report_to_json(report)
        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as f:
                f.write(payload + "\n")
        if not args.quiet:
            print(payload)
        return 0 if report["parallax"]["success"] else 2

    report = verify_spatial_task(
        image_path=str(image_path),
        task=args.task,
        roi=args.roi,
        depth_path=str(depth_path) if depth_path else None,
        table_z=args.table_z,
        config=config,
        isolate=args.isolate,
    )
    payload = report_to_json(report)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            f.write(payload)
            f.write("\n")
    if args.overlay_out:
        write_overlay(str(image_path), report, args.overlay_out, roi=args.roi, config=config)
    if args.robotics_er_prompt_out:
        with open(args.robotics_er_prompt_out, "w", encoding="utf-8") as f:
            f.write(build_robotics_er_prompt(report))
            f.write("\n")
    if args.robotics_er:
        api_key = args.gemini_api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise SystemExit("--robotics-er requires --gemini-api-key or GEMINI_API_KEY")
        analysis = analyze_with_robotics_er(str(image_path), report, api_key=api_key, model=args.robotics_er_model)
        if args.robotics_er_out:
            with open(args.robotics_er_out, "w", encoding="utf-8") as f:
                f.write(analysis)
                f.write("\n")
        elif not args.quiet:
            print("\nROBOTICS-ER ANALYSIS\n")
            print(analysis)
    if not args.quiet:
        print(payload)
    return 0 if report["compliance"]["status"] == "PASS" else 2


def cmd_twoview(args: argparse.Namespace) -> int:
    import json

    from .twoview import check_pair

    res = check_pair(args.view1, args.view2, thresh=args.threshold)
    print(json.dumps(res, indent=2))
    return 0 if res["verdict"] in ("RAISED", "FLAT", "no-object") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spatial-kernel", description="Deterministic, task-relative spatial verification of robot outcomes"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify", help="verify one image against a task profile")
    verify.add_argument("image")
    verify.add_argument("--task", default="bottom_left")
    verify.add_argument("--roi", type=parse_roi)
    verify.add_argument("--depth")
    verify.add_argument("--table-z", type=float)
    verify.add_argument(
        "--isolate",
        default="none",
        choices=["none", "auto", "depth_plane", "largest_component"],
        help="isolate the manipuland before measuring (default none = legacy whole-frame)",
    )
    verify.add_argument(
        "--parallax",
        action="store_true",
        help="success = agreement of independent RGB + depth reads, with Ω as confidence",
    )
    verify.add_argument("--max-side", type=int, default=1536)
    verify.add_argument("--json-out")
    verify.add_argument("--overlay-out")
    verify.add_argument("--robotics-er", action="store_true", help="call Gemini Robotics-ER after deterministic verification")
    verify.add_argument("--robotics-er-model", default="gemini-robotics-er-1.6-preview")
    verify.add_argument("--robotics-er-out")
    verify.add_argument("--robotics-er-prompt-out")
    verify.add_argument("--gemini-api-key")
    verify.add_argument("--quiet", action="store_true")
    verify.set_defaults(func=cmd_verify)
    tv = sub.add_parser("twoview", help="real object or flat print? two photos ~10 cm apart, two black dots on the table")
    tv.add_argument("view1")
    tv.add_argument("view2")
    tv.add_argument("--threshold", type=float, default=15.0, help="relative parallax (px at 1400 wide) that reads RAISED")
    tv.set_defaults(func=cmd_twoview)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

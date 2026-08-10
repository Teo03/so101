#!/usr/bin/env python3
"""Turn a scene observation into a reusable bar-to-basket task program.

This is a geometry/planning handoff only: it writes Cartesian goals and a
skill sequence, but never opens a serial port or moves the physical robot.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perception.workspace_geometry import PlanarWorkspace  # noqa: E402


DEFAULT_SCENE = ROOT / "runtime/outputs/scene_observation.json"
DEFAULT_CALIBRATION = ROOT / "runtime/outputs/current_workspace_calibration.json"
DEFAULT_OUTPUT = ROOT / "runtime/outputs/vision_task_plan.json"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--orientation-mode",
        choices=("proven", "vision"),
        default="vision",
        help=(
            "Use detected object yaw by default. 'proven' keeps the original fixed wrist roll."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene = json.loads(args.scene.read_text())
    workspace = PlanarWorkspace.load(args.calibration)
    bar = scene["selected"]["bar"]
    basket = scene["selected"]["basket"]
    bar_xy = workspace.pixel_to_xy(bar["center_px"])[0]
    basket_xy = workspace.pixel_to_xy(basket["center_px"])[0]
    bar_yaw = workspace.pixel_axis_to_world_yaw(
        bar["center_px"], bar["long_axis_angle_deg"], bar["size_px"][0]
    )
    reference_yaw_deg = workspace.metadata.get("bar_reference_yaw_deg")
    if reference_yaw_deg is None:
        if args.orientation_mode == "vision":
            raise RuntimeError(
                "Vision orientation requires a staged reference yaw in the workspace calibration"
            )
        bar_yaw_delta = 0.0
    else:
        reference_yaw = np.deg2rad(float(reference_yaw_deg))
        bar_yaw_delta = float(
            (bar_yaw - reference_yaw + np.pi / 2.0) % np.pi - np.pi / 2.0
        )
    commanded_yaw_delta = bar_yaw_delta if args.orientation_mode == "vision" else 0.0
    if not (-0.28 <= bar_xy[0] <= 0.02 and -0.14 <= bar_xy[1] <= 0.06):
        raise RuntimeError(f"Detected bar is outside the currently validated pickup region: {bar_xy}")
    if not (-0.02 <= basket_xy[0] <= 0.24 and 0.08 <= basket_xy[1] <= 0.30):
        raise RuntimeError(f"Detected basket is outside the currently validated drop region: {basket_xy}")

    # Aim at the detected basket centre.  The earlier fixed (-8 cm, -6 cm)
    # offset put the released bar visibly to the left of this basket whenever
    # its detected centre moved, defeating the point of visual localization.
    drop_xy = basket_xy.copy()
    # Keep the visual bias inside the independently enforced, physically
    # validated drop envelope.  A basket moved slightly toward the robot can
    # otherwise put the biased point just beyond the near boundary even though
    # the nearest boundary point is still visibly well inside the basket.
    drop_xy = np.clip(
        drop_xy,
        # Stay one millimetre inside the inclusive executor boundary so the
        # six-decimal command serialization cannot round just below it.
        np.asarray((-0.049, 0.061), dtype=np.float64),
        np.asarray((0.16, 0.24), dtype=np.float64),
    )
    task = {
        "format": "so101-task-program-v1",
        "objective": "Pick up the detected chocolate bar and place it in the detected basket.",
        "source_image": scene["image"],
        "calibration": str(args.calibration),
        "orientation_mode": args.orientation_mode,
        "orientation_note": (
            "Detected XY and vision-derived wrist yaw feed standalone IK. The real motor/image "
            "yaw sign was validated through open, close, and lift stages."
        ),
        "objects": {
            "bar": {
                "xy_m": bar_xy.tolist(),
                "z_m": 0.0305,
                "yaw_deg": float(np.degrees(bar_yaw)),
                "yaw_delta_deg": float(np.degrees(bar_yaw_delta)),
                "confidence": bar["confidence"],
            },
            "basket": {
                "xy_m": basket_xy.tolist(),
                "drop_xy_m": drop_xy.tolist(),
                "drop_z_m": 0.105,
                "confidence": basket["confidence"],
            },
        },
        "steps": [
            {"skill": "open_gripper"},
            {"skill": "move_above", "target": "bar", "clearance_m": 0.125},
            {"skill": "align_gripper", "target": "bar", "mode": "pinch_across_short_axis"},
            {"skill": "descend", "target": "bar", "mode": "cartesian_vertical"},
            {"skill": "close_gripper", "mode": "light_preload"},
            {"skill": "lift", "height_m": 0.175},
            {"skill": "move_above", "target": "basket"},
            {"skill": "descend", "target": "basket.drop"},
            {"skill": "open_gripper"},
            {"skill": "retreat"},
            {"skill": "verify", "condition": "bar_inside_basket"},
        ],
        "ik_command": [
            sys.executable,
            str(ROOT / "hardware/plan_pick_place.py"),
            "--pick-x",
            f"{bar_xy[0]:.6f}",
            "--pick-y",
            f"{bar_xy[1]:.6f}",
            "--pick-z",
            "0.028535",
            "--pick-yaw-delta-deg",
            f"{np.degrees(commanded_yaw_delta):.6f}",
            "--drop-x",
            f"{drop_xy[0]:.6f}",
            "--drop-y",
            f"{drop_xy[1]:.6f}",
            "--drop-z",
            "0.105000",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(task, indent=2) + "\n")
    print(json.dumps(task, indent=2), flush=True)
    print(f"[task] wrote dry-run task program to {args.output}", flush=True)


if __name__ == "__main__":
    main()

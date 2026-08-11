#!/usr/bin/env python3
"""Prepare and optionally execute the vision bar-to-basket task.

The default mode is hardware-safe and never opens the follower serial port.
Physical motion additionally
requires both ``--execute`` and the literal ``--confirm MOVE``; the guarded
executor performs the final hardware checks and owns torque teardown.
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE_IMAGE = ROOT / "runtime/outputs/scene_live.png"
DEFAULT_SCENE_JSON = ROOT / "runtime/outputs/scene_observation.json"
DEFAULT_TASK_JSON = ROOT / "runtime/outputs/vision_task_plan.json"
DEFAULT_MOTOR_PLAN = ROOT / "runtime/outputs/vision_pick_place_home.npz"
DEFAULT_POST_IMAGE = ROOT / "runtime/outputs/post_pick_place.png"
DEFAULT_POST_JSON = ROOT / "runtime/outputs/post_pick_place_observation.json"
DEFAULT_EVIDENCE_DIR = Path.home() / "Desktop/so101_vision_pick_place"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="Plan from an existing 640x480 image.")
    parser.add_argument("--camera", default="/dev/video0")
    parser.add_argument("--vision-device", default="0")
    parser.add_argument("--scene-json", type=Path, default=DEFAULT_SCENE_JSON)
    parser.add_argument("--task-json", type=Path, default=DEFAULT_TASK_JSON)
    parser.add_argument("--motor-plan", type=Path, default=DEFAULT_MOTOR_PLAN)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument(
        "--orientation-mode",
        choices=("proven", "vision"),
        default="vision",
        help="Use detected object yaw, or select 'proven' for the original fixed wrist roll.",
    )
    parser.add_argument(
        "--skip-sim-validation",
        action="store_true",
        help="Deprecated compatibility flag; standalone planning never runs simulation.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="After standalone planning, execute the complete real motion.",
    )
    parser.add_argument(
        "--confirm",
        default="",
        help="Physical execution is rejected unless this is exactly MOVE.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.50,
        help="One speed for every task phase, from 0.05 to 1.0.",
    )
    parser.add_argument(
        "--evidence",
        action="store_true",
        help="Pause at important phases to save front/wrist evidence images.",
    )
    return parser.parse_args()


def run(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    child_handles_sigint: bool = False,
) -> None:
    print("[pipeline] " + " ".join(command), flush=True)
    previous_sigint = None
    if child_handles_sigint:
        previous_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        subprocess.run(command, cwd=ROOT, env=env, check=True)
    finally:
        if previous_sigint is not None:
            signal.signal(signal.SIGINT, previous_sigint)


def capture_and_perceive(
    *,
    image: Path | None,
    camera: str,
    output_image: Path,
    output_json: Path,
    annotated: Path,
    device: str,
) -> None:
    attempts = 4 if image is None else 1
    # Never let a failed attempt leave a previously valid observation behind.
    output_json.unlink(missing_ok=True)
    for attempt in range(1, attempts + 1):
        command = [
            sys.executable,
            str(ROOT / "perception/perceive_scene.py"),
            "--device",
            device,
            "--json-output",
            str(output_json),
            "--annotated",
            str(annotated),
        ]
        if image is None:
            command.extend(("--camera", camera, "--capture", str(output_image)))
        else:
            command.extend(("--image", str(image)))
        try:
            run(command)
            return
        except subprocess.CalledProcessError:
            output_json.unlink(missing_ok=True)
            if attempt == attempts:
                raise
            print(
                f"[vision] detection attempt {attempt}/{attempts} was inconclusive; "
                "capturing a fresh frame",
                flush=True,
            )
            time.sleep(0.2)


def validate_task(task: dict[str, object]) -> None:
    if task.get("format") != "so101-task-program-v1":
        raise RuntimeError(f"Unsupported task format: {task.get('format')!r}")
    steps = task.get("steps")
    if not isinstance(steps, list):
        raise RuntimeError("Task program has no step list")
    actual = [step.get("skill") for step in steps if isinstance(step, dict)]
    required_order = [
        "open_gripper",
        "move_above",
        "align_gripper",
        "descend",
        "close_gripper",
        "lift",
        "move_above",
        "descend",
        "open_gripper",
        "retreat",
        "verify",
    ]
    if actual != required_order:
        raise RuntimeError(f"Refusing unexpected task skill sequence: {actual}")
    command = task.get("ik_command")
    if not isinstance(command, list) or not all(isinstance(value, str) for value in command):
        raise RuntimeError("Task program has no valid IK command")
    expected_planner = str(ROOT / "hardware/plan_pick_place.py")
    if len(command) < 2 or command[1] != expected_planner:
        raise RuntimeError(f"Refusing unexpected planner command: {command}")


def verify_postcondition(observation_path: Path) -> bool:
    observation = json.loads(observation_path.read_text())
    selected = observation["selected"]
    bar_center = selected["bar"]["center_px"]
    basket_contour = selected["basket"]["contour_px"]
    # Import lazily so --help and plan inspection do not require OpenCV.
    import cv2
    import numpy as np

    contour = np.asarray(basket_contour, dtype=np.float32).reshape(-1, 1, 2)
    signed_distance = float(cv2.pointPolygonTest(contour, tuple(bar_center), True))
    print(
        f"[verify] bar center={bar_center}; signed distance to basket mask="
        f"{signed_distance:.1f}px",
        flush=True,
    )
    return signed_distance >= 0.0


def main() -> None:
    args = parse_args()
    if not 0.05 <= args.speed <= 1.0:
        raise ValueError("--speed must be between 0.05 and 1.0")
    if args.execute and args.confirm != "MOVE":
        raise RuntimeError("Physical execution requires the literal flag: --confirm MOVE")
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    capture_and_perceive(
        image=args.image,
        camera=args.camera,
        output_image=DEFAULT_SCENE_IMAGE,
        output_json=args.scene_json,
        annotated=args.evidence_dir / "00_planning_annotated.png",
        device=args.vision_device,
    )
    run(
        [
            sys.executable,
            str(ROOT / "code_as_policy/plan_from_scene.py"),
            "--scene",
            str(args.scene_json),
            "--output",
            str(args.task_json),
            "--orientation-mode",
            args.orientation_mode,
        ]
    )
    task = json.loads(args.task_json.read_text())
    validate_task(task)

    planner_command = list(task["ik_command"])
    planner_command.extend(("--output", str(args.motor_plan)))
    run(planner_command)

    if args.execute:
        print(
            "[pipeline] standalone IK plan compiled; real telemetry guards remain active",
            flush=True,
        )

    if not args.execute:
        print(
            f"[pipeline] READY: standalone plan={args.motor_plan} passed numerical checks. "
            "No follower serial port was opened and no simulation was run.",
            flush=True,
        )
        return

    executor_command = [
            sys.executable,
            str(ROOT / "hardware/run_real_cartesian.py"),
            "--plan",
            str(args.motor_plan),
            "--execute-through",
            "postdrop_reset",
            "--speed-scale",
            str(args.speed),
            "--contact-speed-scale",
            str(args.speed),
            "--loaded-speed-scale",
            str(args.speed),
            "--interrupt-return-speed-scale",
            str(args.speed),
            "--confirm",
            "MOVE",
        ]
    if args.evidence:
        executor_command.extend(("--capture-dir", str(args.evidence_dir / "phases")))
    run(
        executor_command,
        child_handles_sigint=True,
    )

    capture_and_perceive(
        image=None,
        camera=args.camera,
        output_image=DEFAULT_POST_IMAGE,
        output_json=DEFAULT_POST_JSON,
        annotated=args.evidence_dir / "99_result_annotated.png",
        device=args.vision_device,
    )
    if not verify_postcondition(DEFAULT_POST_JSON):
        raise RuntimeError(
            "Motion completed safely, but vision did not verify the bar center inside the basket"
        )
    print("[pipeline] SUCCESS: vision verified the bar inside the basket", flush=True)


if __name__ == "__main__":
    main()

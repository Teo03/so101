#!/usr/bin/env python3
"""Safely inspect or execute an offline Cartesian plan on the real SO-101.

The default mode is dry-run and never opens the serial port.  ``--monitor-only``
performs read-only telemetry with torque unchanged.  Physical motion requires
both ``--execute-through`` and the literal ``--confirm MOVE``.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LEROBOT_SRC = ROOT / "lerobot/src"
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus


JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
DEFAULT_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D049262-if00"
DEFAULT_CALIBRATION = (
    Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower/my_follower.json"
)

# Conservative trip thresholds.  These are deliberately below the configured
# servo protection current and are combined with sustained following error so
# one noisy sample does not create a false contact event.
# The first high-clearance trial measured 5.7 degrees of normal elbow lag at
# load 244/current 33 while unfolding.  Keep margin above that observed servo
# acceleration lag; load/current remain independent collision-stop channels.
ARM_ERROR_LIMIT = np.asarray((9.0, 9.0, 9.0, 9.0, 9.0), dtype=np.float32)
GRIPPER_ERROR_LIMIT = 8.0
CURRENT_LIMIT_RAW = np.asarray((140, 140, 140, 120, 100, 100), dtype=np.int32)
LOAD_LIMIT_RAW = np.asarray((450, 450, 450, 400, 350, 300), dtype=np.int32)
# Returning upward against gravity normally produces more following error and
# shoulder load than descending.  The previous safe return measured error 9.37
# and load 436, so use a separate envelope for already-validated reverse motion.
RETURN_ARM_ERROR_LIMIT = np.asarray((14.0, 14.0, 14.0, 14.0, 14.0), dtype=np.float32)
RETURN_CURRENT_LIMIT_RAW = np.asarray((190, 190, 190, 160, 130, 120), dtype=np.int32)
RETURN_LOAD_LIMIT_RAW = np.asarray((600, 650, 600, 520, 450, 380), dtype=np.int32)
# Hard limits are only used during an automatic safety retrace.  Crossing one
# means it is safer to stop driving than to insist on returning to reset.
HARD_CURRENT_LIMIT_RAW = np.asarray((350, 350, 350, 300, 220, 180), dtype=np.int32)
HARD_LOAD_LIMIT_RAW = np.asarray((850, 850, 850, 750, 650, 550), dtype=np.int32)
TEMPERATURE_LIMIT_C = 55
HARD_TEMPERATURE_LIMIT_C = 60
SUSTAINED_TRIP_SAMPLES = 3
# With torque disabled at teardown, wrist flex repeatedly settles near 75 deg
# instead of the 57.6 deg demonstration reset.  The controlled startup already
# interpolates from the measured pose under the normal per-joint speed caps, so
# allow that single, gravity-sensitive joint a wider reset tolerance.  Other
# arm joints retain the tight check to reject arbitrary/collision-prone starts.
START_ERROR_LIMIT = np.asarray((8.0, 8.0, 8.0, 25.0, 8.0, 40.0), dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=ROOT / "runtime/outputs/real_cartesian_plan.npz",
    )
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--monitor-only", action="store_true")
    parser.add_argument("--monitor-seconds", type=float, default=5.0)
    parser.add_argument(
        "--execute-through",
        help="Move through the last sample of this exact plan phase, then stop.",
    )
    parser.add_argument(
        "--recover-to-start",
        action="store_true",
        help="From a held interrupted pose, find the nearest plan sample and retrace to reset.",
    )
    parser.add_argument(
        "--speed-scale",
        type=float,
        default=0.25,
        help="Fraction of planned speed.  0.25 is approximately 4–5 degrees/s.",
    )
    parser.add_argument(
        "--contact-speed-scale",
        type=float,
        help="Optional slower speed for the final descent and gripper closure.",
    )
    parser.add_argument(
        "--loaded-speed-scale",
        type=float,
        help="Optional speed while lifting or lowering a grasped object.",
    )
    parser.add_argument(
        "--confirm",
        default="",
        help="Physical motion is rejected unless this is exactly MOVE.",
    )
    parser.add_argument(
        "--return-to-start",
        action="store_true",
        help="Reverse the validated prefix back to reset before disabling torque.",
    )
    parser.add_argument(
        "--boundary-hold-seconds",
        type=float,
        default=0.0,
        help="Hold and monitor the requested forward boundary before returning.",
    )
    parser.add_argument(
        "--boundary-capture-dir",
        type=Path,
        help="Capture front and wrist 640x480 frames at the forward phase boundary.",
    )
    return parser.parse_args()


def load_calibration(path: Path) -> dict[str, MotorCalibration]:
    payload = json.loads(path.read_text())
    return {name: MotorCalibration(**values) for name, values in payload.items()}


def make_bus(port: str, calibration: dict[str, MotorCalibration]) -> FeetechMotorsBus:
    motors = {
        "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
        "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
        "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
        "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
        "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
        "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
    }
    return FeetechMotorsBus(port=port, motors=motors, calibration=calibration)


def ordered(values: dict[str, float | int], dtype=np.float32) -> np.ndarray:
    return np.asarray([values[name] for name in JOINT_NAMES], dtype=dtype)


def read_telemetry(bus: FeetechMotorsBus) -> dict[str, np.ndarray]:
    return {
        "position": ordered(bus.sync_read("Present_Position"), np.float32),
        "velocity": ordered(bus.sync_read("Present_Velocity", normalize=False), np.int32),
        "load": ordered(bus.sync_read("Present_Load", normalize=False), np.int32),
        "current": ordered(bus.sync_read("Present_Current", normalize=False), np.int32),
        "temperature": ordered(bus.sync_read("Present_Temperature", normalize=False), np.int32),
    }


def hold_and_release(bus: FeetechMotorsBus, reason: str) -> None:
    """Remove commanded pressure, then release torque."""
    try:
        present = bus.sync_read("Present_Position")
        bus.sync_write("Goal_Position", present)
        time.sleep(0.08)
    finally:
        bus.disable_torque(num_retry=3)
    print(f"[real] STOP: {reason}; goal frozen and torque disabled", flush=True)


def start_torque_watchdog(args: argparse.Namespace) -> None:
    """Start a detached fail-safe before any command can enable torque."""
    log_path = Path("/tmp/so101_torque_watchdog.log")
    log = log_path.open("a")
    subprocess.Popen(
        [
            sys.executable,
            str(ROOT / "hardware/real_torque_watchdog.py"),
            "--controller-pid",
            str(os.getpid()),
            "--port",
            args.port,
            "--calibration",
            str(args.calibration),
        ],
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
    log.close()
    print(f"[real] detached torque watchdog armed; log={log_path}", flush=True)


def telemetry_status(
    target: np.ndarray,
    telemetry: dict[str, np.ndarray],
    *,
    returning: bool,
) -> tuple[bool, bool, np.ndarray]:
    """Return soft-trip, hard-trip, and following-error status."""
    error = np.abs(target - telemetry["position"])
    error_limit = RETURN_ARM_ERROR_LIMIT if returning else ARM_ERROR_LIMIT
    current_limit = RETURN_CURRENT_LIMIT_RAW if returning else CURRENT_LIMIT_RAW
    load_limit = RETURN_LOAD_LIMIT_RAW if returning else LOAD_LIMIT_RAW
    soft_trip = bool(
        np.any(error[:5] > error_limit)
        or error[5] > GRIPPER_ERROR_LIMIT
        or np.any(np.abs(telemetry["current"]) > current_limit)
        or np.any(np.abs(telemetry["load"]) > load_limit)
        or np.any(telemetry["temperature"] > TEMPERATURE_LIMIT_C)
    )
    hard_trip = bool(
        np.any(np.abs(telemetry["current"]) > HARD_CURRENT_LIMIT_RAW)
        or np.any(np.abs(telemetry["load"]) > HARD_LOAD_LIMIT_RAW)
        or np.any(telemetry["temperature"] > HARD_TEMPERATURE_LIMIT_C)
    )
    return soft_trip, hard_trip, error


def monitored_retrace(
    bus: FeetechMotorsBus,
    sent_history: list[np.ndarray],
    period: float,
    stop_requested,
) -> tuple[bool, str]:
    """Retrace sent targets to reset while enforcing independent hard limits."""
    if len(sent_history) < 2:
        return False, "no sent path was available to retrace"
    print(
        f"[real] SAFETY RETRACE: reversing {len(sent_history) - 1} already-sent targets",
        flush=True,
    )
    for recovery_index, target in enumerate(reversed(sent_history[:-1])):
        started = time.monotonic()
        if stop_requested():
            return False, "operator interrupted automatic safety retrace"
        bus.sync_write(
            "Goal_Position",
            {
                name: float(value)
                for name, value in zip(JOINT_NAMES, target, strict=True)
            },
        )
        telemetry = read_telemetry(bus)
        _, hard_trip, error = telemetry_status(target, telemetry, returning=True)
        if hard_trip:
            return (
                False,
                "hard telemetry limit during safety retrace: "
                f"error={np.round(error, 2)}, current={telemetry['current']}, "
                f"load={telemetry['load']}, temp={telemetry['temperature']}",
            )
        if recovery_index % 90 == 0:
            print(
                f"[retrace] sample={recovery_index:04d} "
                f"max_error={float(error.max()):.2f} "
                f"current={telemetry['current']} load={telemetry['load']}",
                flush=True,
            )
        remaining = period - (time.monotonic() - started)
        if remaining > 0:
            time.sleep(remaining)
    return True, "automatic safety retrace reached reset"


def monitored_hold(
    bus: FeetechMotorsBus,
    target: np.ndarray,
    duration_s: float,
    stop_requested,
) -> tuple[bool, str]:
    """Hold a phase boundary briefly without turning telemetry monitoring off."""
    deadline = time.monotonic() + max(0.0, duration_s)
    while time.monotonic() < deadline:
        if stop_requested():
            return False, "operator interrupted boundary hold"
        telemetry = read_telemetry(bus)
        _, hard_trip, error = telemetry_status(target, telemetry, returning=True)
        if hard_trip:
            return (
                False,
                "hard telemetry limit during boundary hold: "
                f"error={np.round(error, 2)}, current={telemetry['current']}, "
                f"load={telemetry['load']}, temp={telemetry['temperature']}",
            )
        time.sleep(0.1)
    return True, "boundary hold completed"


def capture_boundary_frames(output_dir: Path, phase: str) -> None:
    """Capture verification frames while the arm is held at a phase boundary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_phase = phase.replace("/", "_")
    for name, camera in (("front", "/dev/video0"), ("wrist", "/dev/video2")):
        output = output_dir / f"{safe_phase}_{name}.png"
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "v4l2",
                "-video_size",
                "640x480",
                "-framerate",
                "30",
                "-i",
                camera,
                "-frames:v",
                "1",
                "-y",
                str(output),
            ],
            check=True,
            timeout=6,
        )
        print(f"[real] captured {name} boundary frame: {output}", flush=True)


def speed_scale_for_phase(
    phase: str,
    *,
    free_space_scale: float,
    contact_scale: float,
    loaded_scale: float,
) -> float:
    """Select a speed without changing any planned joint target."""
    base_phase = phase.removeprefix("return_")
    if base_phase in {"pick_cartesian_08", "close_gripper"}:
        return contact_scale
    if (
        base_phase.startswith("lift_cartesian_")
        or base_phase.startswith("transport_cartesian_")
        or base_phase == "drop"
    ):
        return loaded_scale
    return free_space_scale


def main() -> None:
    args = parse_args()
    if not args.plan.exists():
        raise FileNotFoundError(args.plan)
    plan = np.load(args.plan)
    targets = np.asarray(plan["motor_targets"], dtype=np.float32)
    phases = plan["phase"].astype(str)
    metadata = json.loads(str(plan["metadata_json"]))
    safe_min = np.asarray(metadata["safe_min"], dtype=np.float32)
    safe_max = np.asarray(metadata["safe_max"], dtype=np.float32)
    speed_limit = np.asarray(metadata["speed_limit_per_s"], dtype=np.float32)
    control_hz = float(metadata["control_hz"])
    if targets.ndim != 2 or targets.shape[1] != 6 or len(phases) != len(targets):
        raise RuntimeError("Invalid Cartesian plan shape")
    if np.any(targets < safe_min) or np.any(targets > safe_max):
        raise RuntimeError("Plan violates its declared safety envelope")

    unique_phases = list(dict.fromkeys(phases.tolist()))
    print(
        f"[real] plan={args.plan} samples={len(targets)} duration={len(targets) / control_hz:.1f}s "
        f"phases={unique_phases}",
        flush=True,
    )
    if not args.monitor_only and args.execute_through is None and not args.recover_to_start:
        print("[real] DRY RUN: no serial port opened and no motor command sent", flush=True)
        return

    if not Path(args.port).exists():
        raise FileNotFoundError(f"Follower port not found: {args.port}")
    calibration = load_calibration(args.calibration)
    bus = make_bus(args.port, calibration)
    moving = False
    stop_reason = ""

    def request_stop(signum: int, _frame) -> None:
        nonlocal stop_reason
        stop_reason = f"signal {signum}"

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        bus.connect(handshake=True)
        initial = read_telemetry(bus)
        torque = ordered(bus.sync_read("Torque_Enable", normalize=False), np.int32)
        print(
            f"[real] position={np.round(initial['position'], 2)} torque={torque} "
            f"current={initial['current']} load={initial['load']} temp={initial['temperature']}",
            flush=True,
        )
        if args.monitor_only:
            deadline = time.monotonic() + max(0.1, args.monitor_seconds)
            while time.monotonic() < deadline and not stop_reason:
                telemetry = read_telemetry(bus)
                print(
                    f"[monitor] pos={np.round(telemetry['position'], 2)} "
                    f"current={telemetry['current']} load={telemetry['load']} "
                    f"temp={telemetry['temperature']}",
                    flush=True,
                )
                time.sleep(0.25)
            return

        if args.confirm != "MOVE":
            raise RuntimeError("Physical execution requires the literal flag: --confirm MOVE")
        if args.recover_to_start and args.execute_through is not None:
            raise ValueError("--recover-to-start and --execute-through are mutually exclusive")
        if not args.recover_to_start and args.execute_through not in unique_phases:
            raise ValueError(
                f"Unknown --execute-through phase {args.execute_through!r}; choose one of {unique_phases}"
            )
        if not 0.05 <= args.speed_scale <= 0.5:
            raise ValueError("--speed-scale must be between 0.05 and 0.5")
        contact_speed_scale = (
            args.speed_scale
            if args.contact_speed_scale is None
            else args.contact_speed_scale
        )
        loaded_speed_scale = (
            args.speed_scale
            if args.loaded_speed_scale is None
            else args.loaded_speed_scale
        )
        if not 0.05 <= contact_speed_scale <= 0.5:
            raise ValueError("--contact-speed-scale must be between 0.05 and 0.5")
        if not 0.05 <= loaded_speed_scale <= 0.5:
            raise ValueError("--loaded-speed-scale must be between 0.05 and 0.5")
        if not 0.0 <= args.boundary_hold_seconds <= 30.0:
            raise ValueError("--boundary-hold-seconds must be between 0 and 30")
        if args.recover_to_start:
            normalized_distance = np.max(
                np.abs(targets - initial["position"]) / np.maximum(speed_limit, 1.0),
                axis=1,
            )
            nearest_index = int(np.argmin(normalized_distance))
            nearest_error = np.abs(targets[nearest_index] - initial["position"])
            if np.any(nearest_error > np.asarray((8, 8, 8, 8, 8, 12), dtype=np.float32)):
                raise RuntimeError(
                    "Held pose is not close enough to this plan for automatic recovery: "
                    f"nearest={nearest_index}/{phases[nearest_index]} "
                    f"error={np.round(nearest_error, 2)}"
                )
            command_targets = targets[: nearest_index + 1][::-1].copy()
            command_phases = np.asarray(
                [f"return_{phase}" for phase in phases[: nearest_index + 1][::-1]]
            )
            forward_sample_count = 0
            requested_label = (
                f"recovery from sample {nearest_index}/{phases[nearest_index]}"
            )
            print(
                f"[real] RECOVERY matched sample={nearest_index} "
                f"phase={phases[nearest_index]} error={np.round(nearest_error, 2)}",
                flush=True,
            )
        else:
            stop_indices = np.flatnonzero(phases == args.execute_through)
            stop_index = int(stop_indices[-1])
            command_targets = targets[: stop_index + 1]
            command_phases = phases[: stop_index + 1]
            forward_sample_count = len(command_targets)
            requested_label = str(args.execute_through)
            if args.return_to_start and len(command_targets) > 1:
                command_targets = np.concatenate((command_targets, command_targets[-2::-1]), axis=0)
                command_phases = np.concatenate(
                    (
                        command_phases,
                        np.asarray([f"return_{phase}" for phase in command_phases[-2::-1]]),
                    )
                )
            start_error = np.abs(initial["position"] - command_targets[0])
            if np.any(start_error > START_ERROR_LIMIT):
                raise RuntimeError(
                    f"Arm is not close enough to the planned reset. Error={np.round(start_error, 2)}; "
                    f"limit={START_ERROR_LIMIT}"
                )
            if np.any(torque != 0):
                raise RuntimeError("Expected torque disabled before controlled startup")

        # Prevent a startup jump: write the measured pose as the goal before
        # enabling torque.  Configuration registers are left unchanged.
        start_torque_watchdog(args)
        current_goal = {name: float(value) for name, value in zip(JOINT_NAMES, initial["position"], strict=True)}
        bus.sync_write("Goal_Position", current_goal)
        if np.any(torque == 0):
            if np.any(torque != 0):
                raise RuntimeError(f"Refusing mixed torque state: {torque}")
            bus.enable_torque(num_retry=3)
        moving = True
        # Interpolate from the live pose to the plan's reset target under the
        # same per-joint delta caps.  This is especially important after an
        # emergency stop leaves the gripper open or a joint a few degrees away.
        startup_delta_limit = speed_limit / control_hz
        startup_count = max(
            1,
            int(np.ceil(np.max(np.abs(command_targets[0] - initial["position"]) / startup_delta_limit))),
        )
        startup_targets = np.linspace(
            initial["position"],
            command_targets[0],
            startup_count + 1,
            dtype=np.float32,
        )[1:]
        execution_targets = np.concatenate((startup_targets, command_targets), axis=0)
        execution_phases = np.concatenate(
            (
                np.full(len(startup_targets), "startup_ramp"),
                command_phases,
            )
        )
        forward_boundary_index = (
            len(startup_targets) + forward_sample_count - 1
            if forward_sample_count
            else -1
        )
        trip_count = 0
        previous_phase = ""
        sent_history: list[np.ndarray] = []
        print(
            f"[real] EXECUTING {requested_label}: "
            f"free={args.speed_scale:.2f}x contact={contact_speed_scale:.2f}x "
            f"loaded={loaded_speed_scale:.2f}x. "
            "A forward soft trip automatically retraces; "
            "Ctrl-C freezes and disables torque.",
            flush=True,
        )
        for index, (target, phase) in enumerate(
            zip(execution_targets, execution_phases, strict=True)
        ):
            started = time.monotonic()
            phase_speed_scale = speed_scale_for_phase(
                phase,
                free_space_scale=args.speed_scale,
                contact_scale=contact_speed_scale,
                loaded_scale=loaded_speed_scale,
            )
            period = (1.0 / control_hz) / phase_speed_scale
            if stop_reason:
                hold_and_release(bus, stop_reason)
                moving = False
                return
            target_dict = {
                name: float(value) for name, value in zip(JOINT_NAMES, target, strict=True)
            }
            bus.sync_write("Goal_Position", target_dict)
            sent_history.append(target.copy())
            telemetry = read_telemetry(bus)
            # Both an explicit reverse and the planned empty post-drop path
            # move upward/back toward reset against shoulder gravity.  Use the
            # validated return envelope for each; the first full placement
            # otherwise tripped the tighter forward threshold at 9.72 degrees
            # of harmless shoulder lag after the object had been released.
            returning = phase.startswith("return_") or phase.startswith("postdrop_")
            base_phase = phase.removeprefix("return_")
            loaded_motion = (
                base_phase.startswith("lift_cartesian_")
                or base_phase.startswith("transport_cartesian_")
                or base_phase == "drop"
            )
            soft_trip, hard_trip, error = telemetry_status(
                target,
                telemetry,
                returning=returning or loaded_motion,
            )
            if hard_trip:
                reason = (
                    f"hard telemetry limit at sample {index}: error={np.round(error, 2)}, "
                    f"current={telemetry['current']}, load={telemetry['load']}, "
                    f"temp={telemetry['temperature']}"
                )
                hold_and_release(bus, reason)
                moving = False
                return
            trip_count = trip_count + 1 if soft_trip else 0
            if trip_count >= SUSTAINED_TRIP_SAMPLES and not returning:
                reason = (
                    f"soft contact/stall telemetry at sample {index}: "
                    f"error={np.round(error, 2)}, current={telemetry['current']}, "
                    f"load={telemetry['load']}, temp={telemetry['temperature']}"
                )
                print(f"[real] {reason}", flush=True)
                recovered, recovery_reason = monitored_retrace(
                    bus,
                    sent_history,
                    period,
                    lambda: bool(stop_reason),
                )
                hold_and_release(bus, recovery_reason)
                moving = False
                return
            if trip_count >= SUSTAINED_TRIP_SAMPLES and returning:
                print(
                    f"[real] return-load warning at sample {index}; continuing the "
                    "validated reverse path under hard telemetry limits",
                    flush=True,
                )
                trip_count = 0
            if phase != previous_phase:
                print(
                    f"[real] sample={index:04d} phase={phase:28s} "
                    f"max_error={float(error.max()):.2f} current={telemetry['current']} "
                    f"load={telemetry['load']}",
                    flush=True,
                )
                previous_phase = phase
            if (
                index == forward_boundary_index
                and args.return_to_start
            ):
                if args.boundary_capture_dir is not None:
                    try:
                        capture_boundary_frames(args.boundary_capture_dir, str(args.execute_through))
                    except Exception as error:
                        print(f"[real] boundary capture warning: {error}", flush=True)
                if args.boundary_hold_seconds <= 0:
                    continue
                print(
                    f"[real] holding {args.execute_through} for "
                    f"{args.boundary_hold_seconds:.1f}s before return",
                    flush=True,
                )
                held, hold_reason = monitored_hold(
                    bus,
                    target,
                    args.boundary_hold_seconds,
                    lambda: bool(stop_reason),
                )
                if not held:
                    hold_and_release(bus, hold_reason)
                    moving = False
                    return
            remaining = period - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)

        final = read_telemetry(bus)
        print(
            f"[real] completed through {args.execute_through}"
            f"{' and returned to reset' if args.return_to_start or args.recover_to_start else ''}; "
            f"final={np.round(final['position'], 2)} "
            f"current={final['current']} load={final['load']}",
            flush=True,
        )
        hold_and_release(bus, "requested phase boundary reached")
        moving = False
    finally:
        if bus.is_connected:
            if moving:
                try:
                    hold_and_release(bus, "executor cleanup")
                except Exception as error:
                    print(f"[real] cleanup warning: {error}", flush=True)
            bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()

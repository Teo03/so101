"""Reusable task-to-motor-trajectory compiler for the physical SO-101."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kinematics.so101 import SAFE_MAX as ARM_SAFE_MAX
from kinematics.so101 import SAFE_MIN as ARM_SAFE_MIN
from kinematics.so101 import SO101Kinematics


JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
SAFE_MIN = np.append(ARM_SAFE_MIN, 0.0).astype(np.float32)
SAFE_MAX = np.append(ARM_SAFE_MAX, 88.0).astype(np.float32)
# 85th-percentile per-joint velocities measured from the 35 physically
# demonstrated bar pick/place episodes. This makes speed=1.0 ACT-like without
# using the higher-acceleration tail that the scripted unfold could not follow.
SPEED_LIMIT = np.asarray([24.0, 58.0, 61.0, 32.0, 16.0, 24.0], dtype=np.float32)
CONTROL_HZ = 30.0
RESET = np.asarray([-4.75, -104.84, 96.18, 57.63, 5.32, 0.65], dtype=np.float32)
SAFE_UNFOLD_1 = np.asarray([-6.20, -85.80, 65.10, 70.50, 11.60, 29.93], dtype=np.float32)
SAFE_UNFOLD_2 = np.asarray([-38.30, -5.50, -26.70, 100.00, 1.50, 29.93], dtype=np.float32)
# Use the calibrated mechanical maximum for every approach and release.  A
# partially open jaw reduced the lateral capture window and could leave the
# package outside one fingertip even when the wrist camera showed a small
# localization error.
OPEN_GRIPPER = 88.0
# Firmer than the original 14.3 degree preload, while retaining clearance for
# the package thickness instead of driving against the zero-degree hard stop.
CLOSED_GRIPPER = 8.0
PICK_ROLL = -32.52
DROP_ROLL = 17.01


@dataclass(frozen=True)
class PickPlaceTask:
    pick_xyz_m: tuple[float, float, float]
    drop_xyz_m: tuple[float, float, float]
    transport_z_m: float = 0.175
    return_home: bool = True
    name: str = "pick_place"
    pick_yaw_delta_deg: float = 0.0


def _with_gripper(arm: np.ndarray, gripper: float) -> np.ndarray:
    return np.append(arm[:5], gripper).astype(np.float32)


def _interpolate(waypoints: list[tuple[str, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    samples = [waypoints[0][1].copy()]
    phases = [waypoints[0][0]]
    for (_, previous), (name, target) in zip(waypoints, waypoints[1:], strict=False):
        # Cubic smoothstep has zero velocity at both ends and a peak derivative
        # of 1.5. Account for that factor so peak joint velocity, not merely
        # average velocity, stays inside the demonstrated ACT envelope.
        seconds = 1.5 * float(np.max(np.abs(target - previous) / SPEED_LIMIT))
        count = max(2, int(np.ceil(seconds * CONTROL_HZ)))
        for progress in np.linspace(0.0, 1.0, count + 1, dtype=np.float32)[1:]:
            alpha = progress * progress * (3.0 - 2.0 * progress)
            samples.append((1.0 - alpha) * previous + alpha * target)
            phases.append(name)
    return np.stack(samples).astype(np.float32), np.asarray(phases)


def build_pick_place_plan(task: PickPlaceTask, output: Path) -> Path:
    """Compile a generic pick/place description without opening robot hardware."""
    solver = SO101Kinematics()
    pick = np.asarray(task.pick_xyz_m, dtype=float)
    drop = np.asarray(task.drop_xyz_m, dtype=float)
    above_pick = pick.copy()
    above_drop = drop.copy()
    above_pick[2] = task.transport_z_m
    above_drop[2] = task.transport_z_m

    # Real staging established the motor/image sign: a positive detected bar
    # yaw requires a positive calibrated wrist-roll correction.
    requested_pick_roll = PICK_ROLL + task.pick_yaw_delta_deg
    pick_roll = float(np.clip(requested_pick_roll, ARM_SAFE_MIN[4], ARM_SAFE_MAX[4]))
    applied_yaw_delta = pick_roll - PICK_ROLL
    if not np.isclose(requested_pick_roll, pick_roll):
        print(
            "[plan] wrist orientation saturated at the calibrated joint limit: "
            f"requested_delta={task.pick_yaw_delta_deg:.1f}deg "
            f"applied_delta={applied_yaw_delta:.1f}deg",
            flush=True,
        )
    pick_above_seed = np.asarray([-48.88, 16.59, -46.53, 100.00, pick_roll])
    pick_seed = np.asarray([-48.90, 47.48, -40.85, 87.51, pick_roll])
    drop_seed = np.asarray([1.49, 56.78, -78.45, 75.79, DROP_ROLL])
    descent: list[tuple[str, np.ndarray]] = []
    for index, alpha in enumerate(np.linspace(0.0, 1.0, 9)):
        xyz = (1.0 - alpha) * above_pick + alpha * pick
        seed = (1.0 - alpha) * pick_above_seed + alpha * pick_seed
        seed = solver.inverse_position(xyz, seed, fixed_wrist_roll_deg=pick_roll)
        descent.append((f"pick_cartesian_{index:02d}", _with_gripper(seed, OPEN_GRIPPER)))

    above_drop_q = solver.inverse_position(above_drop, drop_seed, fixed_wrist_roll_deg=DROP_ROLL)
    transport: list[tuple[str, np.ndarray]] = []
    seed = descent[0][1][:5]
    for index, alpha in enumerate(np.linspace(0.125, 1.0, 8), start=1):
        xyz = (1.0 - alpha) * above_pick + alpha * above_drop
        roll = (1.0 - alpha) * pick_roll + alpha * DROP_ROLL
        seed_guess = (1.0 - alpha) * descent[0][1][:5] + alpha * above_drop_q
        seed = solver.inverse_position(xyz, seed_guess, fixed_wrist_roll_deg=float(roll), tolerance_m=0.018)
        transport.append((f"transport_cartesian_{index:02d}", _with_gripper(seed, CLOSED_GRIPPER)))
    drop_q = solver.inverse_position(drop, drop_seed, fixed_wrist_roll_deg=DROP_ROLL)

    sparse: list[tuple[str, np.ndarray]] = [
        # Home is an open-gripper state. This matches the explicit post-drop
        # reset and avoids closing from 88 degrees only to reopen immediately
        # at the start of the next task.
        ("reset", np.append(RESET[:5], OPEN_GRIPPER).astype(np.float32)),
        ("open_gripper", np.append(RESET[:5], OPEN_GRIPPER).astype(np.float32)),
        ("safe_unfold_1", SAFE_UNFOLD_1.copy()),
        ("safe_unfold_2", SAFE_UNFOLD_2.copy()),
        *descent,
        ("close_gripper", _with_gripper(descent[-1][1], CLOSED_GRIPPER)),
    ]
    sparse.extend((name.replace("pick_", "lift_"), _with_gripper(q, CLOSED_GRIPPER)) for name, q in reversed(descent[:-1]))
    sparse.extend(transport)
    sparse.extend(
        [
            ("drop", _with_gripper(drop_q, CLOSED_GRIPPER)),
            ("release", _with_gripper(drop_q, OPEN_GRIPPER)),
            ("retreat", _with_gripper(above_drop_q, OPEN_GRIPPER)),
        ]
    )
    if task.return_home:
        sparse.extend((name.replace("transport_", "postdrop_transport_"), _with_gripper(q, OPEN_GRIPPER)) for name, q in reversed(transport[:-1]))
        sparse.extend(
            [
                ("postdrop_above_pick", _with_gripper(descent[0][1], OPEN_GRIPPER)),
                ("postdrop_safe_unfold_2", SAFE_UNFOLD_2.copy()),
                ("postdrop_safe_unfold_1", SAFE_UNFOLD_1.copy()),
                ("postdrop_reset", np.append(RESET[:5], OPEN_GRIPPER).astype(np.float32)),
            ]
        )

    targets, phases = _interpolate(sparse)
    envelope_tolerance_deg = 0.02
    if np.any(targets < SAFE_MIN - envelope_tolerance_deg) or np.any(
        targets > SAFE_MAX + envelope_tolerance_deg
    ):
        below = np.argwhere(targets < SAFE_MIN - envelope_tolerance_deg)
        above = np.argwhere(targets > SAFE_MAX + envelope_tolerance_deg)
        details: list[str] = []
        for sample, joint in np.concatenate((below, above))[:8]:
            details.append(
                f"{phases[sample]}/{JOINT_NAMES[joint]}="
                f"{targets[sample, joint]:.2f} outside "
                f"[{SAFE_MIN[joint]:.2f}, {SAFE_MAX[joint]:.2f}]"
            )
        raise RuntimeError(
            "Standalone plan escaped the demonstrated joint envelope: " + "; ".join(details)
        )
    targets = np.clip(targets, SAFE_MIN, SAFE_MAX)
    metadata = {
        "format": "so101-real-cartesian-plan-v1",
        "planner": "standalone-placo",
        "task": task.name,
        "control_hz": CONTROL_HZ,
        "joint_names": JOINT_NAMES,
        "pick_xyz_m": pick.tolist(),
        "drop_xyz_m": drop.tolist(),
        "pick_yaw_delta_deg": task.pick_yaw_delta_deg,
        "applied_pick_yaw_delta_deg": applied_yaw_delta,
        "safe_min": SAFE_MIN.tolist(),
        "safe_max": SAFE_MAX.tolist(),
        "speed_limit_per_s": SPEED_LIMIT.tolist(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        motor_targets=targets,
        phase=phases,
        sparse_phase=np.asarray([name for name, _ in sparse]),
        sparse_motor_targets=np.stack([q for _, q in sparse]),
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    return output

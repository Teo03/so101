#!/usr/bin/env python3
"""Plan a demonstrated-branch Cartesian bar pick-and-place for the real SO-101.

This script runs only in Isaac Lab.  It never opens a serial port and cannot
command the physical arm.  The real demonstrations are used only as IK seeds
and safety bounds; the grasp, lift, transport, and drop points are solved from
Cartesian workspace targets in the calibrated digital twin.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=ROOT / "runtime/outputs/real_cartesian_plan.npz")
parser.add_argument("--bar-x", type=float, help="Detected real bar center X in robot-base metres.")
parser.add_argument("--bar-y", type=float, help="Detected real bar center Y in robot-base metres.")
parser.add_argument(
    "--bar-yaw-delta-deg",
    type=float,
    default=0.0,
    help="Detected bar yaw change relative to the successful fixed-pose grasp.",
)
parser.add_argument("--drop-x", type=float, help="Detected basket drop X in robot-base metres.")
parser.add_argument("--drop-y", type=float, help="Detected basket drop Y in robot-base metres.")
parser.add_argument(
    "--return-home-after-drop",
    action="store_true",
    help="Append a collision-cleared, open-gripper return to the reset pose after release.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# The calibrated scene owns two camera sensors even though this planner does
# not read their pixels.  Isaac requires camera rendering to be enabled when
# those sensor prims are present.
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils

from isaac.bridge import lerobot_to_sim_radians, sim_radians_to_lerobot
from isaac.scene import BarPickPlaceScene


JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

# Full observed action range from the 35-episode, two-camera real dataset,
# padded by roughly one degree.  A solved plan outside this envelope is
# rejected instead of silently clamped.
REAL_SAFE_MIN = np.asarray([-66.0, -108.0, -99.0, 38.0, -76.0, 0.0], dtype=np.float32)
REAL_SAFE_MAX = np.asarray([20.0, 73.0, 98.0, 103.0, 49.0, 88.0], dtype=np.float32)

# Conservative command speeds, substantially below the fastest demonstrated
# motion.  Units are LeRobot degrees/s for the arm and normalized units/s for
# the gripper.
REAL_SPEED_LIMIT = np.asarray([18.0, 22.0, 22.0, 18.0, 16.0, 24.0], dtype=np.float32)
CONTROL_HZ = 30.0

# Episode 17 was the closest real episode to the median phase postures across
# 34 automatically segmented successful demonstrations.  These are seeds, not
# a trajectory to replay.
DEMO_SEEDS = {
    "reset": np.asarray([-4.75, -104.84, 96.18, 57.63, 5.32, 0.65], dtype=np.float32),
    "safe_unfold_1": np.asarray([-6.20, -85.80, 65.10, 70.50, 11.60, 0.60], dtype=np.float32),
    "safe_unfold_2": np.asarray([-38.30, -5.50, -26.70, 100.00, 1.50, 24.20], dtype=np.float32),
    "pick_open": np.asarray([-50.11, 37.49, -48.44, 94.64, -9.36, 38.09], dtype=np.float32),
    "pick_closed": np.asarray([-45.98, 54.02, -58.55, 96.57, -3.21, 1.96], dtype=np.float32),
    "pre_release": np.asarray([8.62, 38.11, -96.35, 77.49, 22.11, 1.63], dtype=np.float32),
    "release_open": np.asarray([6.68, 38.02, -96.44, 79.60, 17.01, 23.25], dtype=np.float32),
}

# The true open-gap midpoint measured from the SO-101 collision geometry,
# expressed in the fixed gripper rigid-body frame.  The moving jaw makes this
# 16.3 mm different from the closed-jaw midpoint; using the closed value while
# descending leaves the open fingers visibly close but physically off-centre.
OPEN_PINCH_OFFSET_GRIPPER_M = (0.01629, -0.00024, -0.10325)
CLOSED_PINCH_OFFSET_GRIPPER_M = (0.0, -0.00016, -0.10407)

# Fixed-layout Cartesian targets in the calibrated real/sim workspace.
APPROACH_HEIGHT_M = 0.125
# Put the pinch midpoint at the middle of the 18 mm-thick collision proxy,
# slightly below the rendered scan centre after the object settles.  Targeting
# the top surface only grazes the wrapper and produces no opposing contact.
GRASP_Z_OFFSET_M = -0.002
DROP_XYZ_M = np.asarray((0.04, 0.14, 0.105), dtype=np.float32)
TRANSPORT_HEIGHT_M = 0.175
# Reuse the physically successful cube strategy: 0.40 rad gives a measured
# 48.4 mm open gap.  For this 26 mm-wide bar, 0.12 rad gives approximately a
# 25.8 mm gap, producing light preload instead of the ejection-prone fully
# closed command used in the demonstrations.
BAR_OPEN_JAW_SIM_RAD = 0.40
BAR_GRASP_JAW_SIM_RAD = 0.10


def rotate_vector(quat_wxyz: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    quat_vector = quat_wxyz[:, 1:]
    twice_cross = 2.0 * torch.cross(quat_vector, vector, dim=-1)
    return vector + quat_wxyz[:, :1] * twice_cross + torch.cross(quat_vector, twice_cross, dim=-1)


def interpolate_motor_path(
    named_waypoints: list[tuple[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate sparse motor waypoints at 30 Hz under per-joint speed caps."""
    samples: list[np.ndarray] = [named_waypoints[0][1].copy()]
    phases: list[str] = [named_waypoints[0][0]]
    for (previous_name, previous), (name, target) in zip(named_waypoints, named_waypoints[1:], strict=False):
        del previous_name
        required_seconds = float(np.max(np.abs(target - previous) / REAL_SPEED_LIMIT))
        # One second minimum makes every phase observable and avoids abrupt
        # contact commands even when only the jaw changes.
        count = max(2, int(np.ceil(max(1.0, required_seconds) * CONTROL_HZ)))
        for alpha in np.linspace(0.0, 1.0, count + 1, dtype=np.float32)[1:]:
            samples.append((1.0 - alpha) * previous + alpha * target)
            phases.append(name)
    return np.stack(samples).astype(np.float32), np.asarray(phases)


def main() -> None:
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / 120.0, render_interval=8, device=args_cli.device)
    )
    scene = BarPickPlaceScene(sim, item="bar")
    sim.reset()
    scene.reset()
    for _ in range(4):
        scene.step(scene.robot.data.default_joint_pos.clone())

    from isaaclab.sim.utils import get_current_stage
    from pxr import Usd, UsdGeom

    stage = get_current_stage()
    bar_prim = stage.GetPrimAtPath("/World/envs/env_0/Bar")
    bounds = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    ).ComputeWorldBound(bar_prim).ComputeAlignedBox()
    bar_center = 0.5 * (
        np.asarray(bounds.GetMin(), dtype=np.float32) + np.asarray(bounds.GetMax(), dtype=np.float32)
    )
    if (args_cli.bar_x is None) != (args_cli.bar_y is None):
        raise ValueError("--bar-x and --bar-y must be supplied together")
    if args_cli.bar_x is not None:
        detected_xy = np.asarray((args_cli.bar_x, args_cli.bar_y), dtype=np.float32)
        nominal_xy = np.asarray((-0.190, -0.050), dtype=np.float32)
        if np.any(np.abs(detected_xy - nominal_xy) > np.asarray((0.055, 0.055))):
            raise RuntimeError(
                f"Detected bar XY is outside the validated local region: {detected_xy}"
            )
        bar_center[:2] = detected_xy
    if abs(args_cli.bar_yaw_delta_deg) > 40.0:
        raise RuntimeError(
            f"Detected bar yaw change is outside the validated range: "
            f"{args_cli.bar_yaw_delta_deg:.1f}deg"
        )
    if (args_cli.drop_x is None) != (args_cli.drop_y is None):
        raise ValueError("--drop-x and --drop-y must be supplied together")
    drop_xyz = DROP_XYZ_M.copy()
    if args_cli.drop_x is not None:
        drop_xyz[:2] = (args_cli.drop_x, args_cli.drop_y)
        if not (-0.05 <= drop_xyz[0] <= 0.16 and 0.06 <= drop_xyz[1] <= 0.24):
            raise RuntimeError(f"Detected drop target is outside the validated region: {drop_xyz[:2]}")
    collision_prim = stage.GetPrimAtPath("/World/envs/env_0/Bar/collision_proxy")
    collision_bounds = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy]
    ).ComputeWorldBound(collision_prim).ComputeAlignedBox()
    collision_min = np.asarray(collision_bounds.GetMin(), dtype=np.float32)
    collision_max = np.asarray(collision_bounds.GetMax(), dtype=np.float32)
    collision_center = 0.5 * (collision_min + collision_max)
    print(
        f"[plan] bar visual center={np.round(bar_center, 4)} "
        f"collision center={np.round(collision_center, 4)} "
        f"collision size={np.round(collision_max - collision_min, 4)}",
        flush=True,
    )

    robot = scene.robot
    gripper_body_ids, _ = robot.find_bodies("gripper")
    arm_joint_ids, arm_joint_names = robot.find_joints("^(Rotation|Pitch|Elbow|Wrist_Pitch|Wrist_Roll)$")
    jaw_ids, _ = robot.find_joints("Jaw")
    if len(gripper_body_ids) != 1 or len(arm_joint_ids) != 5 or len(jaw_ids) != 1:
        raise RuntimeError("Unexpected SO-101 articulation layout")
    ee_body_id = gripper_body_ids[0]
    ee_jacobian_id = ee_body_id - 1 if robot.is_fixed_base else ee_body_id
    wrist_roll_arm_index = arm_joint_names.index("Wrist_Roll")
    wrist_roll_joint_id = arm_joint_ids[wrist_roll_arm_index]
    open_pinch_offset_b = torch.tensor(
        [OPEN_PINCH_OFFSET_GRIPPER_M], dtype=torch.float32, device=scene.scene.device
    )
    closed_pinch_offset_b = torch.tensor(
        [CLOSED_PINCH_OFFSET_GRIPPER_M], dtype=torch.float32, device=scene.scene.device
    )

    full_lower = torch.as_tensor(
        lerobot_to_sim_radians(REAL_SAFE_MIN), dtype=torch.float32, device=scene.scene.device
    ).unsqueeze(0)
    full_upper = torch.as_tensor(
        lerobot_to_sim_radians(REAL_SAFE_MAX), dtype=torch.float32, device=scene.scene.device
    ).unsqueeze(0)
    arm_lower = torch.maximum(robot.data.joint_limits[:, arm_joint_ids, 0], full_lower[:, arm_joint_ids])
    arm_upper = torch.minimum(robot.data.joint_limits[:, arm_joint_ids, 1], full_upper[:, arm_joint_ids])

    def configuration_from_real(real_values: np.ndarray) -> torch.Tensor:
        configuration = robot.data.default_joint_pos.clone()
        configuration[0] = torch.as_tensor(
            lerobot_to_sim_radians(real_values), dtype=torch.float32, device=scene.scene.device
        )
        configuration[:, arm_joint_ids] = torch.clamp(
            configuration[:, arm_joint_ids], arm_lower, arm_upper
        )
        return configuration

    def write_configuration(configuration: torch.Tensor) -> None:
        robot.write_joint_state_to_sim(configuration, torch.zeros_like(configuration))
        robot.set_joint_position_target(configuration)
        scene.scene.write_data_to_sim()
        sim.step()
        scene.scene.update(sim.get_physics_dt())

    def pinch_position(offset_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        offset_w = rotate_vector(robot.data.body_quat_w[:, ee_body_id], offset_b)
        return robot.data.body_pos_w[:, ee_body_id] + offset_w, offset_w

    def opening_axis_angle() -> float:
        """Jaw-opening heading in world XY, measured from live gripper geometry."""
        local_axis = torch.tensor([[1.0, 0.0, 0.0]], device=scene.scene.device)
        axis_w = rotate_vector(robot.data.body_quat_w[:, ee_body_id], local_axis)[0]
        return float(torch.atan2(axis_w[1], axis_w[0]).item())

    def wrap_angle(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    def solve(
        name: str,
        xyz: np.ndarray,
        seed_real: np.ndarray,
        jaw_real: float,
        *,
        fixed_wrist_real: float | None = None,
        tolerance: float = 0.012,
    ) -> torch.Tensor:
        """Constrained damped-least-squares IK on the demonstrated branch."""
        seed_values = seed_real.copy()
        seed_values[5] = jaw_real
        candidate = configuration_from_real(seed_values)
        if fixed_wrist_real is not None:
            fixed_sim = float(lerobot_to_sim_radians(
                np.asarray((0, 0, 0, 0, fixed_wrist_real, jaw_real), dtype=np.float32)
            )[4])
            candidate[:, wrist_roll_joint_id] = fixed_sim
            active_joint_ids = [joint_id for joint_id in arm_joint_ids if joint_id != wrist_roll_joint_id]
        else:
            active_joint_ids = arm_joint_ids
        # Open descent is centred with the measured open-gap midpoint.  Once
        # the jaw is closed around the object, transport and drop use the
        # measured closed-jaw midpoint.
        point_offset_b = open_pinch_offset_b if jaw_real > 20.0 else closed_pinch_offset_b
        active_lower = robot.data.joint_limits[:, active_joint_ids, 0]
        active_upper = robot.data.joint_limits[:, active_joint_ids, 1]
        active_lower = torch.maximum(active_lower, full_lower[:, active_joint_ids])
        active_upper = torch.minimum(active_upper, full_upper[:, active_joint_ids])
        goal = torch.as_tensor(xyz, dtype=torch.float32, device=scene.scene.device).unsqueeze(0)
        write_configuration(candidate)
        best = candidate.clone()
        best_error = float("inf")
        for _ in range(160):
            point, offset_w = pinch_position(point_offset_b)
            error = goal - point
            distance = float(torch.linalg.vector_norm(error, dim=1).item())
            if distance < best_error:
                best_error = distance
                best = candidate.clone()
            if distance < 0.004:
                break
            body_jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobian_id, :, active_joint_ids]
            linear_jacobian = body_jacobian[:, :3, :]
            angular_columns = body_jacobian[:, 3:6, :].transpose(1, 2)
            point_columns = torch.cross(
                angular_columns, offset_w[:, None, :].expand_as(angular_columns), dim=-1
            )
            jacobian = linear_jacobian + point_columns.transpose(1, 2)
            jt = jacobian.transpose(1, 2)
            damping = 0.025**2 * torch.eye(3, device=scene.scene.device).unsqueeze(0)
            delta = jt @ torch.linalg.solve(jacobian @ jt + damping, error.unsqueeze(-1))
            candidate[:, active_joint_ids] = torch.clamp(
                candidate[:, active_joint_ids] + delta.squeeze(-1).clamp(-0.045, 0.045),
                active_lower,
                active_upper,
            )
            write_configuration(candidate)
        write_configuration(best)
        motor = sim_radians_to_lerobot(best[0].detach().cpu().numpy())
        print(
            f"[plan] {name:24s} error={best_error:.4f} m "
            f"pinch={np.round(pinch_position(point_offset_b)[0][0].detach().cpu().numpy(), 4)} "
            f"motor={np.round(motor, 2)}",
            flush=True,
        )
        if best_error > tolerance:
            raise RuntimeError(f"{name} is not reachable on the demonstrated branch ({best_error:.3f} m)")
        if np.any(motor < REAL_SAFE_MIN - 0.05) or np.any(motor > REAL_SAFE_MAX + 0.05):
            raise RuntimeError(f"{name} escaped the demonstrated safety envelope: {motor}")
        return best

    jaw_conversion = robot.data.default_joint_pos[0].detach().cpu().numpy()
    jaw_conversion[5] = BAR_OPEN_JAW_SIM_RAD
    open_jaw = float(sim_radians_to_lerobot(jaw_conversion)[5])
    jaw_conversion[5] = BAR_GRASP_JAW_SIM_RAD
    closed_jaw = float(sim_radians_to_lerobot(jaw_conversion)[5])
    release_jaw = open_jaw
    pick_roll = float(DEMO_SEEDS["pick_open"][4])
    release_roll = float(DEMO_SEEDS["release_open"][4])

    grasp_xyz = bar_center.copy()
    grasp_xyz[2] += GRASP_Z_OFFSET_M
    above_pick_xyz = grasp_xyz.copy()
    above_pick_xyz[2] = max(TRANSPORT_HEIGHT_M, grasp_xyz[2] + APPROACH_HEIGHT_M)
    above_drop_xyz = drop_xyz.copy()
    above_drop_xyz[2] = TRANSPORT_HEIGHT_M

    grasp_q = solve(
        "grasp_open",
        grasp_xyz,
        DEMO_SEEDS["pick_open"],
        open_jaw,
        fixed_wrist_real=pick_roll,
    )
    # The scanned bar collision proxy is 122 mm along world X and 26 mm along
    # world Y.  The jaws must therefore open along world Y.  Measure the live
    # wrist response instead of assuming that real and USD roll zeroes match.
    write_configuration(grasp_q)
    opening_before = opening_axis_angle()
    probe_q = grasp_q.clone()
    probe_q[:, wrist_roll_joint_id] += 0.08
    probe_q[:, wrist_roll_joint_id] = torch.clamp(
        probe_q[:, wrist_roll_joint_id],
        full_lower[:, wrist_roll_joint_id],
        full_upper[:, wrist_roll_joint_id],
    )
    actual_probe = float((probe_q[0, wrist_roll_joint_id] - grasp_q[0, wrist_roll_joint_id]).item())
    write_configuration(probe_q)
    opening_after = opening_axis_angle()
    response = wrap_angle(opening_after - opening_before) / actual_probe
    bar_yaw_delta = np.deg2rad(args_cli.bar_yaw_delta_deg)
    # Image yaw and the gripper opening heading have opposite signs under the
    # real camera/robot convention.  The first vision trial added this delta
    # and visibly rotated the fingers away from the bar.  Subtracting the
    # measured delta keeps the opening perpendicular to the moved bar.
    candidate_axes = (0.5 * np.pi - bar_yaw_delta, -0.5 * np.pi - bar_yaw_delta)
    desired_opening = min(candidate_axes, key=lambda angle: abs(wrap_angle(angle - opening_before)))
    required_sim_delta = wrap_angle(desired_opening - opening_before) / response
    aligned_q = grasp_q.clone()
    aligned_q[:, wrist_roll_joint_id] = torch.clamp(
        aligned_q[:, wrist_roll_joint_id] + required_sim_delta,
        full_lower[:, wrist_roll_joint_id],
        full_upper[:, wrist_roll_joint_id],
    )
    aligned_real = sim_radians_to_lerobot(aligned_q[0].detach().cpu().numpy())
    pick_roll = float(aligned_real[4])
    print(
        f"[plan] opening axis={np.degrees(opening_before):.1f} deg "
        f"response={response:.3f}; align to {np.degrees(desired_opening):.1f} deg "
        f"with real wrist_roll={pick_roll:.2f}",
        flush=True,
    )
    grasp_q = solve(
        "grasp_open_aligned",
        grasp_xyz,
        aligned_real,
        open_jaw,
        fixed_wrist_real=pick_roll,
    )
    grasp_real = sim_radians_to_lerobot(grasp_q[0].detach().cpu().numpy())

    descent: list[tuple[str, np.ndarray]] = []
    previous_real = grasp_real.copy()
    for index, alpha in reversed(list(enumerate(np.linspace(0.0, 1.0, 9), start=0))):
        xyz = (1.0 - alpha) * above_pick_xyz + alpha * grasp_xyz
        q = solve(
            f"pick_cartesian_{index:02d}",
            xyz,
            previous_real,
            open_jaw,
            fixed_wrist_real=pick_roll,
        )
        previous_real = sim_radians_to_lerobot(q[0].detach().cpu().numpy())
        descent.append((f"pick_cartesian_{index:02d}", previous_real.copy()))
    # The loop solved grasp-to-above for branch continuity; execution needs the
    # reverse order (above-to-grasp).
    descent.reverse()

    above_drop_q = solve(
        "above_drop",
        above_drop_xyz,
        DEMO_SEEDS["pre_release"],
        closed_jaw,
        fixed_wrist_real=release_roll,
    )
    above_drop_real = sim_radians_to_lerobot(above_drop_q[0].detach().cpu().numpy())
    drop_q = solve(
        "drop",
        drop_xyz,
        DEMO_SEEDS["pre_release"],
        closed_jaw,
        fixed_wrist_real=release_roll,
    )
    drop_real = sim_radians_to_lerobot(drop_q[0].detach().cpu().numpy())

    # Solve high Cartesian transport points using a seed interpolated between
    # independently demonstrated pick and release branches.
    above_pick_real = descent[0][1].copy()
    transport: list[tuple[str, np.ndarray]] = []
    for index, alpha in enumerate(np.linspace(0.125, 1.0, 8), start=1):
        xyz = (1.0 - alpha) * above_pick_xyz + alpha * above_drop_xyz
        seed = (1.0 - alpha) * above_pick_real + alpha * above_drop_real
        roll = (1.0 - alpha) * pick_roll + alpha * release_roll
        q = solve(
            f"transport_cartesian_{index:02d}",
            xyz,
            seed,
            closed_jaw,
            fixed_wrist_real=float(roll),
            tolerance=0.018,
        )
        transport.append(
            (f"transport_cartesian_{index:02d}", sim_radians_to_lerobot(q[0].detach().cpu().numpy()))
        )

    sparse: list[tuple[str, np.ndarray]] = [
        ("reset", DEMO_SEEDS["reset"].copy()),
        ("open_gripper", np.concatenate((DEMO_SEEDS["reset"][:5], [open_jaw])).astype(np.float32)),
        (
            "safe_unfold_1",
            np.concatenate((DEMO_SEEDS["safe_unfold_1"][:5], [open_jaw])).astype(np.float32),
        ),
        (
            "safe_unfold_2",
            np.concatenate((DEMO_SEEDS["safe_unfold_2"][:5], [open_jaw])).astype(np.float32),
        ),
        *descent,
    ]
    close_real = descent[-1][1].copy()
    close_real[5] = closed_jaw
    sparse.append(("close_gripper", close_real))
    for name, values in reversed(descent[:-1]):
        lift = values.copy()
        lift[5] = closed_jaw
        sparse.append((name.replace("pick_", "lift_"), lift))
    sparse.extend(transport)
    sparse.append(("drop", drop_real))
    release_real = drop_real.copy()
    release_real[5] = release_jaw
    sparse.append(("release", release_real))
    retreat_real = above_drop_real.copy()
    retreat_real[5] = release_jaw
    sparse.append(("retreat", retreat_real))
    if args_cli.return_home_after_drop:
        # Return over the same high Cartesian corridor with the gripper open.
        # This avoids reversing the whole plan, which would unnecessarily
        # close the empty gripper again in the basket and at the source.
        for name, values in reversed(transport[:-1]):
            homeward = values.copy()
            homeward[5] = release_jaw
            sparse.append((name.replace("transport_", "postdrop_transport_"), homeward))
        postdrop_above_pick = descent[0][1].copy()
        postdrop_above_pick[5] = release_jaw
        sparse.append(("postdrop_above_pick", postdrop_above_pick))
        sparse.append(
            (
                "postdrop_safe_unfold_2",
                np.concatenate((DEMO_SEEDS["safe_unfold_2"][:5], [release_jaw])).astype(np.float32),
            )
        )
        sparse.append(
            (
                "postdrop_safe_unfold_1",
                np.concatenate((DEMO_SEEDS["safe_unfold_1"][:5], [release_jaw])).astype(np.float32),
            )
        )
        postdrop_reset = DEMO_SEEDS["reset"].copy()
        postdrop_reset[5] = release_jaw
        sparse.append(("postdrop_reset", postdrop_reset))

    motor_targets, phases = interpolate_motor_path(sparse)
    delta = np.abs(np.diff(motor_targets, axis=0))
    per_frame_limit = REAL_SPEED_LIMIT / CONTROL_HZ + 1e-5
    if np.any(delta > per_frame_limit):
        raise RuntimeError(
            f"Interpolation exceeded per-frame speed limit: {delta.max(axis=0)} > {per_frame_limit}"
        )
    if np.any(motor_targets < REAL_SAFE_MIN) or np.any(motor_targets > REAL_SAFE_MAX):
        raise RuntimeError("Interpolated plan escaped the demonstrated safety envelope")

    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "so101-real-cartesian-plan-v1",
        "purpose": "offline plan; no physical execution authorization",
        "control_hz": CONTROL_HZ,
        "joint_names": JOINT_NAMES,
        "bar_visual_center_m": bar_center.tolist(),
        "bar_yaw_delta_deg": args_cli.bar_yaw_delta_deg,
        "grasp_xyz_m": grasp_xyz.tolist(),
        "drop_xyz_m": drop_xyz.tolist(),
        "return_home_after_drop": bool(args_cli.return_home_after_drop),
        "safe_min": REAL_SAFE_MIN.tolist(),
        "safe_max": REAL_SAFE_MAX.tolist(),
        "speed_limit_per_s": REAL_SPEED_LIMIT.tolist(),
    }
    np.savez_compressed(
        args_cli.output,
        motor_targets=motor_targets,
        phase=phases,
        sparse_phase=np.asarray([name for name, _ in sparse]),
        sparse_motor_targets=np.stack([values for _, values in sparse]).astype(np.float32),
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    print(
        f"[plan] wrote {len(motor_targets)} samples ({len(motor_targets) / CONTROL_HZ:.1f} s) "
        f"to {args_cli.output}",
        flush=True,
    )
    simulation_app.close()


if __name__ == "__main__":
    main()

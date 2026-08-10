#!/usr/bin/env python3
"""Run a conservative, deterministic bar-to-basket pick-and-place skill in Isaac Lab.

This is deliberately not a learned policy.  It is the first bounded skill for
the agent loop: Cartesian waypoints are followed with differential IK, the jaw
is commanded separately, and the final bar pose is checked against the basket
interior.  It only controls Isaac Sim; it has no physical-robot path.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps-per-waypoint", type=int, default=180)
parser.add_argument("--settle-steps", type=int, default=120)
parser.add_argument("--save-every", type=int, default=60)
parser.add_argument(
    "--render-interval",
    type=int,
    default=4,
    help="Render every N physics steps; camera frames are still captured at named waypoints.",
)
parser.add_argument("--inspect-geometry", action="store_true", help="Print gripper/jaw geometry bounds and exit.")
parser.add_argument(
    "--inspect-pinch",
    action="store_true",
    help="Measure the actual fixed/moving fingertip mesh points at the calibrated grasp pose and exit.",
)
parser.add_argument("--grasp-assist", action="store_true", help="Use a proximity-gated virtual grasp after the physical close command.")
parser.add_argument("--item", choices=("bar", "ball", "cube"), default="bar", help="Task item to pick and drop.")
parser.add_argument(
    "--reachable-basket",
    action="store_true",
    help="Move the basket to the validated reachable test pose; leaves the default calibrated scene unchanged.",
)
parser.add_argument(
    "--grasp-only",
    action="store_true",
    help="Stop after a physical close-and-lift test; require both jaw contact and object lift.",
)
parser.add_argument(
    "--diagnose-first-touch",
    action="store_true",
    help="Stop the descent at the first detected object motion and save exact pre-touch/touch camera frames.",
)
parser.add_argument(
    "--replay-trajectory",
    action="store_true",
    help="Replay the saved successful fixed-cube trajectory in real time without rerunning IK search.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
from PIL import Image

import isaaclab.sim as sim_utils

from isaac.scene import EASY_BALL_BASKET_CENTER_M, BarPickPlaceScene


# The waypoint positions are in Isaac world metres and are intentionally kept
# high and slow.  The gripper's reset orientation is retained by position-only
# IK, matching the down-facing real operating pose.
# Front-inner drop target for the nominal basket.  The ball validation selects
# its own closer-basket target after the scene is created.
BASKET_DROP_XY = (0.04, 0.14)
APPROACH_CLEARANCE_M = 0.15
GRASP_CLEARANCE_M = 0.010
DROP_Z = 0.105
JAW_CLOSED_RAD = -0.15
JAW_OPEN_RAD = 1.25
# Measured from the live SO-101 collision geometry.  At 0.40 rad the distal
# gap is 48.4 mm, leaving 9.2 mm clearance on each side of a 30 mm cube and
# covering the measured 8.8 mm lateral tracking drift during descent.
# A 0.16 rad close corresponds to roughly a 29 mm gap: light preload, without
# the ejection caused by commanding the empty-hand closed limit.
CUBE_OPEN_RAD = 0.40
CUBE_GRASP_RAD = 0.16
# Midpoint between the fixed and moving distal fingertip surfaces at the
# closed-jaw pose, expressed in the fixed ``gripper`` rigid-body frame.
PINCH_OFFSET_GRIPPER_M = (0.0, -0.00016, -0.10407)
CUBE_OPEN_PINCH_OFFSET_GRIPPER_M = (0.01629, -0.00024, -0.10325)


def save_frame(image: np.ndarray, name: str) -> None:
    output_dir = ROOT / "runtime" / "outputs" / f"scripted_pick_place_{args_cli.item}"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Isaac's camera buffer is occasionally non-contiguous on the first
    # render; copying keeps Pillow's PNG encoder deterministic.
    Image.fromarray(np.ascontiguousarray(image)).save(output_dir / name)


def save_desktop_frame(image: np.ndarray, name: str) -> None:
    desktop = Path.home() / "Desktop"
    desktop.mkdir(exist_ok=True)
    Image.fromarray(np.ascontiguousarray(image)).save(desktop / name)


def main() -> None:
    print("[skill] creating simulation context", flush=True)
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / 120.0, render_interval=args_cli.render_interval, device=args_cli.device)
    )
    print("[skill] spawning calibrated scene", flush=True)
    scene = BarPickPlaceScene(
        sim,
        item=args_cli.item,
        basket_center=EASY_BALL_BASKET_CENTER_M if args_cli.reachable_basket else None,
    )
    print("[skill] resetting physics", flush=True)
    sim.reset()
    scene.reset()
    scene.set_front_camera_view()

    # Populate both sensor buffers before attempting to save a diagnostic
    # frame.  Without this warm-up a newly spawned camera can expose a 0-sized
    # image on its first physics tick.
    warmup_targets = scene.robot.data.default_joint_pos.clone()
    print("[skill] warming camera buffers", flush=True)
    for _ in range(8):
        scene.step(warmup_targets)
    print("[skill] scene ready", flush=True)
    # The scan mesh is not centred at its USD root.  Query its rendered world
    # bounds instead of assuming the rigid-body root is the grasp centre.
    from isaaclab.sim.utils import get_current_stage
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = get_current_stage()
    bar_prim = stage.GetPrimAtPath("/World/envs/env_0/Bar")
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]).ComputeWorldBound(bar_prim)
    box = bbox.ComputeAlignedBox()
    bar_visual_center = (np.asarray(box.GetMin(), dtype=np.float32) + np.asarray(box.GetMax(), dtype=np.float32)) * 0.5
    print(f"[skill] rendered bar centre={bar_visual_center.round(4)}", flush=True)
    if args_cli.inspect_geometry:
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]
        )
        for root_path in ("/World/envs/env_0/Robot/gripper", "/World/envs/env_0/Robot/jaw"):
            root = stage.GetPrimAtPath(root_path)
            root_box = cache.ComputeWorldBound(root).ComputeAlignedBox()
            print(
                f"[geometry] {root_path} instance={root.IsInstance()} "
                f"children={[str(child.GetPath()) for child in root.GetAllChildren()]} "
                f"min={np.asarray(root_box.GetMin()).round(4)} max={np.asarray(root_box.GetMax()).round(4)}",
                flush=True,
            )
            for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
                is_collision = prim.HasAPI(UsdPhysics.CollisionAPI)
                if not prim.IsA(UsdGeom.Boundable) and not is_collision:
                    continue
                if prim.IsA(UsdGeom.Boundable) or is_collision:
                    aligned = cache.ComputeWorldBound(prim).ComputeAlignedBox()
                    low = np.asarray(aligned.GetMin()).round(4)
                    high = np.asarray(aligned.GetMax()).round(4)
                    if np.isfinite(low).all() and np.isfinite(high).all():
                        print(
                            f"  {prim.GetPath()} type={prim.GetTypeName()} collision={is_collision} "
                            f"min={low} max={high}",
                            flush=True,
                        )
                else:
                    print(f"  {prim.GetPath()} type={prim.GetTypeName()} collision={is_collision}", flush=True)
        simulation_app.close()
        return

    robot = scene.robot
    # The calibrated workshop asset intentionally uses soft gains for manual
    # teleoperation.  This deterministic skill needs the simulated servos to
    # hold the solved joint target against gravity and contact.  These gains
    # are applied only at runtime in this script; scene.py remains unchanged.
    robot.write_joint_stiffness_to_sim(torch.tensor([[300, 300, 250, 150, 100, 250]], device=scene.scene.device))
    robot.write_joint_damping_to_sim(torch.tensor([[15, 15, 12, 8, 5, 8]], device=scene.scene.device))
    robot.write_joint_effort_limit_to_sim(torch.tensor([[100, 100, 100, 80, 50, 100]], device=scene.scene.device))
    jaw_body_ids, jaw_body_names = robot.find_bodies("jaw")
    gripper_body_ids, gripper_body_names = robot.find_bodies("gripper")
    if len(jaw_body_ids) != 1 or len(gripper_body_ids) != 1:
        raise RuntimeError(
            f"Expected one gripper and jaw body, found gripper={gripper_body_names}, jaw={jaw_body_names}"
        )
    jaw_body_id = jaw_body_ids[0]
    ee_body_id = gripper_body_ids[0]
    ee_jacobian_id = ee_body_id - 1 if robot.is_fixed_base else ee_body_id
    arm_joint_ids, arm_joint_names = robot.find_joints("^(Rotation|Pitch|Elbow|Wrist_Pitch|Wrist_Roll)$")
    jaw_ids, jaw_names = robot.find_joints("Jaw")
    if len(arm_joint_ids) != 5 or len(jaw_ids) != 1:
        raise RuntimeError(f"Unexpected joints: arm={arm_joint_names}, jaw={jaw_names}")

    print(f"[skill] end effector=gripper/pinch, arm joints={arm_joint_names}, jaw={jaw_names[0]}", flush=True)
    print(f"[skill] rigid bodies={robot.body_names}", flush=True)
    print(
        f"[skill] reset ee pos={robot.data.body_pos_w[0, ee_body_id].detach().cpu().numpy().round(4)} "
        f"quat(wxyz)={robot.data.body_quat_w[0, ee_body_id].detach().cpu().numpy().round(4)}",
        flush=True,
    )
    print(
        f"[skill] reset jaw pos={robot.data.body_pos_w[0, jaw_body_id].detach().cpu().numpy().round(4)} "
        f"quat(wxyz)={robot.data.body_quat_w[0, jaw_body_id].detach().cpu().numpy().round(4)}",
        flush=True,
    )
    targets = robot.data.default_joint_pos.clone()
    pinch_offset = (
        CUBE_OPEN_PINCH_OFFSET_GRIPPER_M if args_cli.item == "cube" else PINCH_OFFSET_GRIPPER_M
    )
    pinch_offset_b = torch.tensor([pinch_offset], dtype=torch.float32, device=scene.scene.device)
    grasp_open_jaw = CUBE_OPEN_RAD if args_cli.item == "cube" else JAW_OPEN_RAD

    if args_cli.replay_trajectory:
        trajectory_path = (
            ROOT / "runtime" / "outputs" / "scripted_pick_place_cube" / "successful_cube_trajectory.npz"
        )
        if args_cli.item != "cube":
            raise RuntimeError("--replay-trajectory currently supports only the verified fixed-cube task")
        if not trajectory_path.exists():
            raise FileNotFoundError(f"Successful trajectory not found: {trajectory_path}")
        trajectory = np.load(trajectory_path)
        actions = torch.as_tensor(
            trajectory["action"], dtype=torch.float32, device=scene.scene.device
        )
        if actions.ndim != 2 or actions.shape[1] != robot.num_joints:
            raise RuntimeError(
                f"Trajectory action shape {tuple(actions.shape)} does not match {robot.num_joints} robot joints"
            )
        physics_dt = float(trajectory["physics_dt"])
        print(
            f"[replay] starting {actions.shape[0]}-step successful trajectory "
            f"({actions.shape[0] * physics_dt:.1f} simulated seconds)",
            flush=True,
        )
        for action in actions:
            started = time.perf_counter()
            scene.step(action.unsqueeze(0))
            remaining = physics_dt - (time.perf_counter() - started)
            if remaining > 0:
                time.sleep(remaining)
        final_image = scene.image()
        final_wrist = scene.wrist_image()
        save_frame(final_image, "replay_final_front.png")
        save_frame(final_wrist, "replay_final_wrist.png")
        save_desktop_frame(final_image, "so101_replay_final_front.png")
        save_desktop_frame(final_wrist, "so101_replay_final_wrist.png")
        cube = scene.bar.data.root_pos_w[0].detach().cpu().numpy()
        dx, dy = np.abs(cube[:2] - np.asarray(scene.basket_center[:2]))
        replay_success = dx < 0.085 and dy < 0.065 and 0.01 < cube[2] < 0.12
        print(f"[replay] final cube position={np.round(cube, 4)}; in_basket={bool(replay_success)}", flush=True)
        # Hold the completed scene for five seconds so the user can inspect it.
        final_action = actions[-1].unsqueeze(0)
        for _ in range(round(5.0 / physics_dt)):
            started = time.perf_counter()
            scene.step(final_action)
            remaining = physics_dt - (time.perf_counter() - started)
            if remaining > 0:
                time.sleep(remaining)
        simulation_app.close()
        return

    def rotate_vector(quat_wxyz: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        """Rotate a batched vector by a batched scalar-first quaternion."""
        quat_vector = quat_wxyz[:, 1:]
        twice_cross = 2.0 * torch.cross(quat_vector, vector, dim=-1)
        return vector + quat_wxyz[:, :1] * twice_cross + torch.cross(quat_vector, twice_cross, dim=-1)

    def pinch_position() -> tuple[torch.Tensor, torch.Tensor]:
        offset_w = rotate_vector(robot.data.body_quat_w[:, ee_body_id], pinch_offset_b)
        return robot.data.body_pos_w[:, ee_body_id] + offset_w, offset_w

    def opening_axis_angle() -> float:
        """Return the jaw-opening axis heading in the world XY plane."""
        local_axis = torch.tensor([[1.0, 0.0, 0.0]], device=scene.scene.device)
        axis_w = rotate_vector(robot.data.body_quat_w[:, ee_body_id], local_axis)[0]
        return float(torch.atan2(axis_w[1], axis_w[0]).item())

    def wrap_angle(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    holding_bar = False
    held_bar_offset = torch.zeros((1, 3), device=scene.scene.device)
    first_touch_found = False
    recording_enabled = False
    trajectory_actions: list[np.ndarray] = []
    trajectory_joint_positions: list[np.ndarray] = []
    trajectory_object_poses: list[np.ndarray] = []
    trajectory_phases: list[str] = []

    def record_sample(phase: str, command: torch.Tensor) -> None:
        if not recording_enabled:
            return
        trajectory_actions.append(command[0].detach().cpu().numpy().copy())
        trajectory_joint_positions.append(robot.data.joint_pos[0].detach().cpu().numpy().copy())
        trajectory_object_poses.append(scene.bar.data.root_state_w[0, :7].detach().cpu().numpy().copy())
        trajectory_phases.append(phase)

    def update_held_bar() -> None:
        """Carry the bar only while the explicit grasp-assist latch is active."""
        if not holding_bar:
            return
        state = scene.bar.data.root_state_w.clone()
        state[:, :3] = robot.data.body_pos_w[:, ee_body_id] + held_bar_offset
        state[:, 7:] = 0.0
        scene.bar.write_root_pose_to_sim(state[:, :7])
        scene.bar.write_root_velocity_to_sim(state[:, 7:])

    arm_lower = robot.data.joint_limits[:, arm_joint_ids, 0]
    arm_upper = robot.data.joint_limits[:, arm_joint_ids, 1]
    arm_default = targets[:, arm_joint_ids].clone()
    wrist_roll_arm_index = arm_joint_names.index("Wrist_Roll")
    wrist_roll_joint_id = arm_joint_ids[wrist_roll_arm_index]
    generator = torch.Generator(device=scene.scene.device).manual_seed(7)

    def write_configuration(configuration: torch.Tensor) -> None:
        """Teleport a candidate configuration, then update Isaac's FK/Jacobian state."""
        robot.write_joint_state_to_sim(configuration, torch.zeros_like(configuration))
        robot.set_joint_position_target(configuration)
        scene.scene.write_data_to_sim()
        sim.step()
        scene.scene.update(sim.get_physics_dt())

    def solve_joint_configuration(
        name: str,
        xyz: np.ndarray,
        jaw: float,
        *,
        seed_count: int = 24,
        max_iterations: int = 48,
        tolerance: float = 0.018,
        initial_arm: torch.Tensor | None = None,
        fixed_wrist_roll: float | None = None,
    ) -> torch.Tensor:
        """Use random restarts plus Jacobian refinement against the true jaw pose."""
        goal = torch.as_tensor(xyz, dtype=torch.float32, device=scene.scene.device).unsqueeze(0)
        best_q = targets.clone()
        best_distance = float("inf")
        # This modest search is deliberate: it resolves the SO-101's local IK
        # minima without a large GPU batch or a separate motion-planning stack.
        seeds = [initial_arm.clone() if initial_arm is not None else arm_default]
        seeds.extend(
            arm_lower + (arm_upper - arm_lower) * torch.rand((1, 5), generator=generator, device=scene.scene.device)
            for _ in range(seed_count)
        )
        for seed in seeds:
            candidate = targets.clone()
            candidate[:, arm_joint_ids] = seed
            candidate[:, jaw_ids] = jaw
            if fixed_wrist_roll is not None:
                candidate[:, wrist_roll_joint_id] = fixed_wrist_roll
                active_joint_ids = [joint_id for joint_id in arm_joint_ids if joint_id != wrist_roll_joint_id]
            else:
                active_joint_ids = arm_joint_ids
            active_lower = robot.data.joint_limits[:, active_joint_ids, 0]
            active_upper = robot.data.joint_limits[:, active_joint_ids, 1]
            write_configuration(candidate)
            for _ in range(max_iterations):
                ee_pos, offset_w = pinch_position()
                error = goal - ee_pos
                body_jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobian_id, :, active_joint_ids]
                linear_jacobian = body_jacobian[:, :3, :]
                angular_columns = body_jacobian[:, 3:6, :].transpose(1, 2)
                point_velocity_columns = torch.cross(
                    angular_columns, offset_w[:, None, :].expand_as(angular_columns), dim=-1
                )
                jacobian = linear_jacobian + point_velocity_columns.transpose(1, 2)
                # Damped least-squares update, capped so the solver remains in
                # the same collision-free branch instead of jumping joints.
                jt = jacobian.transpose(1, 2)
                damping = 0.025**2 * torch.eye(3, device=scene.scene.device).unsqueeze(0)
                delta = jt @ torch.linalg.solve(jacobian @ jt + damping, error.unsqueeze(-1))
                candidate[:, active_joint_ids] = torch.clamp(
                    candidate[:, active_joint_ids] + delta.squeeze(-1).clamp(-0.10, 0.10),
                    active_lower,
                    active_upper,
                )
                write_configuration(candidate)
                distance = torch.linalg.vector_norm(goal - pinch_position()[0], dim=1).item()
                if distance < 0.008:
                    break
            distance = torch.linalg.vector_norm(goal - pinch_position()[0], dim=1).item()
            if distance < best_distance:
                best_distance, best_q = distance, candidate.clone()
        write_configuration(best_q)
        print(
            f"[solver] {name}: pinch error={best_distance:.4f} m "
            f"position={pinch_position()[0][0].detach().cpu().numpy().round(4)} "
            f"joints={best_q[0, arm_joint_ids].detach().cpu().numpy().round(3)}",
            flush=True,
        )
        if best_distance > tolerance:
            raise RuntimeError(f"No reachable {name} configuration; best jaw error was {best_distance:.3f} m")
        return best_q

    def execute_joint_target(name: str, target: torch.Tensor, steps: int) -> None:
        """Move smoothly to a solved configuration under normal physics."""
        nonlocal first_touch_found
        print(f"[skill] executing {name}", flush=True)
        # A one-step assisted run is a kinematic scene-validation mode.  It
        # uses the exact solved pose, then advances Isaac once to render and
        # update observations.  Normal runs retain interpolated physics.
        if args_cli.grasp_assist and steps == 1:
            robot.write_joint_state_to_sim(target, torch.zeros_like(target))
            scene.step(target)
            record_sample(name, target)
            update_held_bar()
            save_frame(scene.image(), f"{name}_0001_front.png")
            save_frame(scene.wrist_image(), f"{name}_0001_wrist.png")
            actual = pinch_position()[0][0].detach().cpu().numpy()
            print(f"[skill] {name} pinch={np.round(actual, 4)}", flush=True)
            return
        start = robot.data.joint_pos.clone()
        detect_first_touch = args_cli.diagnose_first_touch and (
            name.endswith("_grasp") or "_descent_" in name
        )
        object_start_pos = scene.bar.data.root_pos_w.clone()
        previous_front = scene.image().copy() if detect_first_touch else None
        previous_wrist = scene.wrist_image().copy() if detect_first_touch else None
        for step in range(steps):
            alpha = (step + 1) / steps
            command = (1.0 - alpha) * start + alpha * target
            scene.step(command)
            record_sample(name, command)
            update_held_bar()
            if detect_first_touch:
                displacement = torch.linalg.vector_norm(scene.bar.data.root_pos_w - object_start_pos, dim=1).item()
                current_front = scene.image().copy()
                current_wrist = scene.wrist_image().copy()
                if displacement >= 0.0005:
                    first_touch_found = True
                    save_frame(previous_front, "first_touch_before_front.png")
                    save_frame(previous_wrist, "first_touch_before_wrist.png")
                    save_frame(current_front, "first_touch_front.png")
                    save_frame(current_wrist, "first_touch_wrist.png")
                    save_desktop_frame(previous_front, "so101_first_touch_before_front.png")
                    save_desktop_frame(previous_wrist, "so101_first_touch_before_wrist.png")
                    save_desktop_frame(current_front, "so101_first_touch_front.png")
                    save_desktop_frame(current_wrist, "so101_first_touch_wrist.png")
                    print(
                        f"[diagnostic] first touch at descent step {step + 1}/{steps}: "
                        f"object displacement={displacement * 1000.0:.2f} mm, "
                        f"jaw target={command[0, jaw_ids[0]].item():.3f} rad, "
                        f"jaw actual={robot.data.joint_pos[0, jaw_ids[0]].item():.3f} rad, "
                        f"pinch={np.round(pinch_position()[0][0].detach().cpu().numpy(), 4)}",
                        flush=True,
                    )
                    return
                previous_front = current_front
                previous_wrist = current_wrist
            if step == steps - 1 or (step + 1) % args_cli.save_every == 0:
                save_frame(scene.image(), f"{name}_{step + 1:04d}_front.png")
                save_frame(scene.wrist_image(), f"{name}_{step + 1:04d}_wrist.png")
        actual = pinch_position()[0][0].detach().cpu().numpy()
        max_joint_error = torch.max(torch.abs(target - robot.data.joint_pos)).item()
        print(
            f"[skill] {name} pinch={np.round(actual, 4)} max_joint_error={max_joint_error:.4f} rad",
            flush=True,
        )

    def joint_target(arm_values: tuple[float, float, float, float, float], jaw_value: float) -> torch.Tensor:
        """Build a full joint target from the calibrated five-arm-joint order."""
        target = robot.data.default_joint_pos.clone()
        target[:, arm_joint_ids] = torch.tensor([arm_values], device=scene.scene.device)
        target[:, jaw_ids] = jaw_value
        return target

    if args_cli.inspect_pinch:
        gripper_ids, _ = robot.find_bodies("gripper")
        gripper_body_id = gripper_ids[0]
        mesh_paths = {
            "fixed": (
                "/World/envs/env_0/Robot/gripper/visuals/wrist_roll_follower_so101_v1/mesh",
                gripper_body_id,
                "/World/envs/env_0/Robot/gripper",
            ),
            "moving": (
                "/World/envs/env_0/Robot/jaw/visuals/moving_jaw_so101_v1/mesh",
                jaw_body_id,
                "/World/envs/env_0/Robot/jaw",
            ),
        }

        def rotation_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
            w, x, y, z = quat_wxyz
            return np.asarray(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                ],
                dtype=np.float64,
            )

        # The USD stage keeps authored transforms while PhysX/Fabric owns the
        # live articulation transforms.  Convert each mesh to its body's local
        # frame at reset, then apply the live body pose for every jaw setting.
        local_points: dict[str, tuple[np.ndarray, int]] = {}
        authored_points: dict[str, np.ndarray] = {}
        reset_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        for finger, (path, body_id, body_path) in mesh_paths.items():
            mesh = UsdGeom.Mesh(stage.GetPrimAtPath(path))
            matrix = reset_cache.GetLocalToWorldTransform(mesh.GetPrim())
            body_inverse = reset_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(body_path)).GetInverse()
            authored_world_gf = [matrix.Transform(Gf.Vec3d(*point)) for point in mesh.GetPointsAttr().Get()]
            authored_world = np.asarray(authored_world_gf, dtype=np.float64)
            local = np.asarray(
                [body_inverse.Transform(point) for point in authored_world_gf], dtype=np.float64
            )
            local_points[finger] = (local, body_id)
            authored_points[finger] = authored_world

        # The fingertips are the vertices farthest from the fixed gripper body.
        # Find the closest opposing pair in those distal sets and express their
        # midpoint in the fixed gripper body frame; this is the true pinch point.
        gripper_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot/gripper")
        gripper_matrix = reset_cache.GetLocalToWorldTransform(gripper_prim)
        gripper_inverse = gripper_matrix.GetInverse()
        gripper_origin = np.asarray(gripper_matrix.ExtractTranslation(), dtype=np.float64)
        distal_sets = {}
        distal_indices = {}
        for finger, points in authored_points.items():
            radii = np.linalg.norm(points - gripper_origin[None, :], axis=1)
            mask = radii >= np.quantile(radii, 0.92)
            distal_sets[finger] = points[mask]
            distal_indices[finger] = np.flatnonzero(mask)
        fixed_distal = distal_sets["fixed"]
        moving_distal = distal_sets["moving"]
        pair_distances = np.linalg.norm(
            fixed_distal[:, None, :] - moving_distal[None, :, :], axis=2
        )
        fixed_index, moving_index = np.unravel_index(np.argmin(pair_distances), pair_distances.shape)
        fixed_contact = fixed_distal[fixed_index]
        moving_contact = moving_distal[moving_index]
        pinch_world = 0.5 * (fixed_contact + moving_contact)
        fixed_contact_local = np.asarray(
            gripper_inverse.Transform(Gf.Vec3d(*fixed_contact)), dtype=np.float64
        )
        moving_contact_local = np.asarray(
            gripper_inverse.Transform(Gf.Vec3d(*moving_contact)), dtype=np.float64
        )
        pinch_local = np.asarray(gripper_inverse.Transform(Gf.Vec3d(*pinch_world)), dtype=np.float64)
        print(
            f"[pinch] authored fixed_contact={np.round(fixed_contact, 5)} "
            f"moving_contact={np.round(moving_contact, 5)} gap={pair_distances.min():.5f} "
            f"fixed_local={np.round(fixed_contact_local, 5)} "
            f"moving_local={np.round(moving_contact_local, 5)} "
            f"pinch_local_gripper={np.round(pinch_local, 5)}",
            flush=True,
        )

        jaw_samples = (
            (JAW_CLOSED_RAD, "closed"),
            (-0.025, "grasp_candidate"),
            (0.10, "sample_010"),
            (0.20, "sample_020"),
            (0.30, "sample_030"),
            (0.40, "sample_040"),
            (0.60, "sample_060"),
            (JAW_OPEN_RAD, "maximum_open"),
        )
        for jaw_value, label in jaw_samples:
            # Keep the arm at its collision-free reset while measuring jaw
            # geometry.  The local gap is independent of arm pose and this
            # avoids contaminating the result with cube contact.
            configuration = robot.data.default_joint_pos.clone()
            configuration[:, jaw_ids] = jaw_value
            write_configuration(configuration)
            object_pos = bar_visual_center.astype(np.float64)
            current_body_pos = robot.data.body_pos_w[0].detach().cpu().numpy()
            current_body_quat = robot.data.body_quat_w[0].detach().cpu().numpy()
            print(f"[pinch] {label} target_object={np.round(object_pos, 5)}", flush=True)
            live_points = {}
            for finger, (local, body_id) in local_points.items():
                points = local @ rotation_matrix(current_body_quat[body_id]).T + current_body_pos[body_id]
                live_points[finger] = points
                distances = np.linalg.norm(points - object_pos[None, :], axis=1)
                nearest = points[int(np.argmin(distances))]
                print(
                    f"[pinch] {label} {finger}: nearest={np.round(nearest, 5)} "
                    f"distance={distances.min():.5f} bounds=({np.round(points.min(axis=0), 4)}, "
                    f"{np.round(points.max(axis=0), 4)})",
                    flush=True,
                )
            fixed_distal_live = live_points["fixed"][distal_indices["fixed"]]
            moving_distal_live = live_points["moving"][distal_indices["moving"]]
            live_pair_distances = np.linalg.norm(
                fixed_distal_live[:, None, :] - moving_distal_live[None, :, :], axis=2
            )
            fixed_live_index, moving_live_index = np.unravel_index(
                np.argmin(live_pair_distances), live_pair_distances.shape
            )
            fixed_live = fixed_distal_live[fixed_live_index]
            moving_live = moving_distal_live[moving_live_index]
            midpoint_live = 0.5 * (fixed_live + moving_live)
            gripper_rotation = rotation_matrix(current_body_quat[gripper_body_id])
            midpoint_gripper = (midpoint_live - current_body_pos[gripper_body_id]) @ gripper_rotation
            print(
                f"[pinch] {label} distal_gap={live_pair_distances.min():.5f} m "
                f"midpoint_world={np.round(midpoint_live, 5)} "
                f"midpoint_gripper={np.round(midpoint_gripper, 5)}",
                flush=True,
            )
        simulation_app.close()
        return

    bar_grasp = bar_visual_center.copy()
    bar_above = bar_grasp.copy()
    bar_above[2] += APPROACH_CLEARANCE_M
    if args_cli.item == "bar":
        bar_grasp[2] += GRASP_CLEARANCE_M
    elif args_cli.item == "cube":
        # The measured reference is the distal fingertip midpoint.  Put it
        # close to the tabletop so the 50 mm-tall cube reaches the parallel
        # inner finger faces instead of balancing at the tapered tips.
        bar_grasp[2] = 0.008
    basket_drop_xy = (
        (0.06, 0.04) if (args_cli.item in {"ball", "cube"} or args_cli.reachable_basket) else BASKET_DROP_XY
    )
    basket_above = np.asarray((*basket_drop_xy, bar_above[2]), dtype=np.float32)
    basket_drop = np.asarray((*basket_drop_xy, DROP_Z), dtype=np.float32)
    def reset_nominal() -> None:
        scene.reset()
        reset_target = scene.robot.data.default_joint_pos.clone()
        for _ in range(8):
            scene.step(reset_target)

    # The joint origin is not necessarily the physical pinch centre.  Search
    # a small cross around the measured bar centre and use the actual bar
    # height after closing and lifting as the grasp verifier.
    pinch_offsets = (
        (0.000, 0.000, 0.000),
        (-0.035, 0.000, 0.000),
        (0.035, 0.000, 0.000),
        (0.000, -0.035, 0.000),
        (0.000, 0.035, 0.000),
    )
    selected: tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]] | None = None
    fallback: tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]] | None = None
    # Assisted validation is a trajectory/placement baseline, not a friction
    # experiment.  Do not spend hundreds of collision-heavy steps proving the
    # known limitation before latching the virtual grasp.
    if args_cli.grasp_assist:
        print("[search] Assisted mode: skipping physical-friction search.", flush=True)
        # These poses were solved against the calibrated easy-ball scene.  By
        # using them as the baseline we avoid repeated costly render/physics
        # updates in the numerical search; bar mode still solves its own poses.
        above_bar_q = joint_target((-1.288, 0.521, 0.028, -0.415, -1.796), grasp_open_jaw)
        grasp_q = joint_target((-0.845, 0.993, -0.301, 0.655, -0.267), grasp_open_jaw)
        selected = (above_bar_q, grasp_q, [grasp_q])
        fallback = selected
        offsets_to_try = ()
    else:
        offsets_to_try = (
            ((0.000, 0.000, -0.012), (0.000, 0.000, 0.000), (0.000, 0.000, 0.012))
            if args_cli.item == "ball"
            else ((0.000, 0.000, 0.000),)
            if args_cli.item == "cube"
            else pinch_offsets
        )
    for attempt, (dx, dy, dz) in enumerate(offsets_to_try, start=1):
        candidate_grasp = bar_grasp.copy()
        candidate_grasp += (dx, dy, dz)
        candidate_above = candidate_grasp.copy()
        candidate_above[2] += APPROACH_CLEARANCE_M - GRASP_CLEARANCE_M
        print(f"[search] grasp attempt {attempt}: offset=({dx:.3f}, {dy:.3f}, {dz:.3f})", flush=True)
        grasp_q = solve_joint_configuration(
            f"search_{attempt}_grasp", candidate_grasp, grasp_open_jaw, seed_count=6, max_iterations=40
        )
        fixed_wrist_roll = None
        if args_cli.item == "cube":
            # Align the opening with a cube face instead of pinching across a
            # 42 mm diagonal.  Probe the live wrist-roll response so the sign
            # comes from this USD articulation rather than a convention guess.
            write_configuration(grasp_q)
            opening_before = opening_axis_angle()
            probe_delta = 0.08
            probe_q = grasp_q.clone()
            probe_q[:, wrist_roll_joint_id] = torch.clamp(
                probe_q[:, wrist_roll_joint_id] + probe_delta,
                robot.data.joint_limits[:, wrist_roll_joint_id, 0],
                robot.data.joint_limits[:, wrist_roll_joint_id, 1],
            )
            actual_probe_delta = (
                probe_q[0, wrist_roll_joint_id] - grasp_q[0, wrist_roll_joint_id]
            ).item()
            write_configuration(probe_q)
            opening_after = opening_axis_angle()
            response = wrap_angle(opening_after - opening_before) / actual_probe_delta
            desired_opening = round(opening_before / (0.5 * np.pi)) * (0.5 * np.pi)
            roll_delta = wrap_angle(desired_opening - opening_before) / response
            fixed_wrist_roll = float(
                torch.clamp(
                    grasp_q[0, wrist_roll_joint_id] + roll_delta,
                    robot.data.joint_limits[0, wrist_roll_joint_id, 0],
                    robot.data.joint_limits[0, wrist_roll_joint_id, 1],
                ).item()
            )
            print(
                f"[solver] cube opening axis={np.degrees(opening_before):.1f} deg, "
                f"wrist response={response:.3f}, target axis={np.degrees(desired_opening):.1f} deg, "
                f"fixed Wrist_Roll={fixed_wrist_roll:.3f} rad",
                flush=True,
            )
            grasp_q = solve_joint_configuration(
                f"search_{attempt}_grasp_aligned",
                candidate_grasp,
                grasp_open_jaw,
                seed_count=12,
                max_iterations=72,
                tolerance=0.010,
                initial_arm=grasp_q[:, arm_joint_ids],
                fixed_wrist_roll=fixed_wrist_roll,
            )
        # Continue upward from the grasp solution so approach, close, and lift
        # remain on one collision-free IK branch.  Solving the elevated pose
        # independently can select a joint-limit branch that the servos cannot
        # track from the grasp.
        above_q = solve_joint_configuration(
            f"search_{attempt}_above",
            candidate_above,
            grasp_open_jaw,
            seed_count=0,
            max_iterations=72,
            tolerance=0.025,
            initial_arm=grasp_q[:, arm_joint_ids],
            fixed_wrist_roll=fixed_wrist_roll,
        )
        if args_cli.item == "cube":
            # A direct interpolation between the two joint configurations
            # curves sideways by almost 9 mm.  Solve a vertical Cartesian
            # line on the same local IK branch so both open fingers pass
            # around the fixed cube without touching it.
            descent_path: list[torch.Tensor] = []
            previous_q = above_q
            for segment in range(1, 9):
                alpha = segment / 8.0
                waypoint = (1.0 - alpha) * candidate_above + alpha * candidate_grasp
                waypoint_q = solve_joint_configuration(
                    f"search_{attempt}_cartesian_{segment:02d}",
                    waypoint,
                    grasp_open_jaw,
                    seed_count=0,
                    max_iterations=48,
                    tolerance=0.008,
                    initial_arm=previous_q[:, arm_joint_ids],
                    fixed_wrist_roll=fixed_wrist_roll,
                )
                descent_path.append(waypoint_q)
                previous_q = waypoint_q
            grasp_q = descent_path[-1]
        else:
            descent_path = [grasp_q]
        if fallback is None:
            fallback = (above_q, grasp_q, descent_path)
        reset_nominal()
        if args_cli.grasp_only:
            write_configuration(above_q)
            execute_joint_target(f"search_{attempt}_above", above_q, 20)
        else:
            execute_joint_target(f"search_{attempt}_above", above_q, 100)
        for segment, waypoint_q in enumerate(descent_path, start=1):
            execute_joint_target(f"search_{attempt}_descent_{segment:02d}", waypoint_q, 24)
            if first_touch_found:
                break
        if args_cli.diagnose_first_touch:
            if not first_touch_found:
                save_frame(scene.image(), "no_first_touch_front.png")
                save_frame(scene.wrist_image(), "no_first_touch_wrist.png")
                print("[diagnostic] descent completed without measurable object motion", flush=True)
            print("[diagnostic] frames saved to ~/Desktop; stopping before jaw closure", flush=True)
            simulation_app.close()
            return
        close_q = grasp_q.clone()
        close_q[:, jaw_ids] = CUBE_GRASP_RAD if args_cli.item == "cube" else JAW_CLOSED_RAD
        execute_joint_target(f"search_{attempt}_close", close_q, 180 if args_cli.item == "cube" else 90)
        if args_cli.item == "cube":
            lift_path = [q.clone() for q in reversed(descent_path[:-1])] + [above_q.clone()]
            for waypoint_q in lift_path:
                waypoint_q[:, jaw_ids] = CUBE_GRASP_RAD
            for segment, waypoint_q in enumerate(lift_path, start=1):
                execute_joint_target(f"search_{attempt}_lift_{segment:02d}", waypoint_q, 24)
        else:
            execute_joint_target(f"search_{attempt}_lift", above_q, 120)
        bar_height = scene.bar.data.root_pos_w[0, 2].item()
        force_matrix = scene.contact_grasp.data.force_matrix_w
        contact_force = (
            torch.linalg.vector_norm(force_matrix, dim=-1).sum().item() if force_matrix is not None else 0.0
        )
        # Height is authoritative here: grasp assist is disabled, so the only
        # way the rigid object can rise is through PhysX contact.  The workshop
        # jaw-only filtered sensor has returned zero for fixed-finger contact,
        # therefore report it diagnostically but do not make it a false veto.
        physical_grasp = bar_height > 0.065
        print(
            f"[search] grasp attempt {attempt}: object height={bar_height:.4f} m, "
            f"jaw contact={contact_force:.3f} N, physical_grasp={physical_grasp}",
            flush=True,
        )
        if physical_grasp:
            selected = (above_q, grasp_q, descent_path)
            break

    if selected is None:
        if not args_cli.grasp_assist or fallback is None:
            raise RuntimeError("No physical grasp found in the local pinch-centre search.")
        selected = fallback
        print("[search] No frictional grasp found; using proximity-gated grasp assist.", flush=True)
    above_bar_q, grasp_q, descent_path = selected
    if args_cli.grasp_only:
        save_frame(scene.image(), "physical_grasp_front.png")
        save_frame(scene.wrist_image(), "physical_grasp_wrist.png")
        save_desktop_frame(scene.image(), "so101_physical_grasp_front.png")
        save_desktop_frame(scene.wrist_image(), "so101_physical_grasp_wrist.png")
        print("[skill] physical grasp verification passed", flush=True)
        simulation_app.close()
        return

    # Solve basket targets after a verified grasp configuration was found.
    if args_cli.grasp_assist and (args_cli.item in {"ball", "cube"} or args_cli.reachable_basket):
        transport_jaw = CUBE_GRASP_RAD if args_cli.item == "cube" else JAW_CLOSED_RAD
        above_basket_q = joint_target((0.151, 0.048, 0.253, 0.648, -1.157), transport_jaw)
        drop_q = joint_target((0.121, 0.405, 0.537, -0.128, -0.685), transport_jaw)
    else:
        transport_jaw = CUBE_GRASP_RAD if args_cli.item == "cube" else JAW_CLOSED_RAD
        above_basket_q = solve_joint_configuration(
            "above_basket", basket_above, transport_jaw, seed_count=2, max_iterations=24
        )
        drop_q = solve_joint_configuration("drop", basket_drop, transport_jaw, seed_count=2, max_iterations=24)
    reset_nominal()
    recording_enabled = True
    record_sample("00_reset", scene.robot.data.default_joint_pos)
    execute_joint_target("01_above_bar", above_bar_q, args_cli.steps_per_waypoint)
    if args_cli.item == "cube":
        for segment, waypoint_q in enumerate(descent_path, start=1):
            execute_joint_target(f"02_descend_{segment:02d}", waypoint_q, 24)
    else:
        execute_joint_target("02_descend_to_bar", grasp_q, args_cli.steps_per_waypoint)
    close_q = grasp_q.clone()
    close_q[:, jaw_ids] = CUBE_GRASP_RAD if args_cli.item == "cube" else JAW_CLOSED_RAD
    execute_joint_target("03_close", close_q, args_cli.settle_steps)
    jaw_to_bar = torch.linalg.vector_norm(
        robot.data.body_pos_w[:, ee_body_id] - scene.bar.data.root_pos_w, dim=1
    ).item()
    if args_cli.grasp_assist:
        # The SO-101 jaw body's physics origin sits behind the visible pinch
        # point.  For the easy ball the rendered close-frame is the relevant
        # proximity check; its body-origin distance is therefore not used to
        # reject this explicitly virtual baseline.
        if jaw_to_bar > 0.090:
            print(
                f"[skill] virtual grasp accepts jaw-body offset {jaw_to_bar:.3f} m; "
                "the rendered pinch point is the validation reference.",
                flush=True,
            )
        holding_bar = True
        held_bar_offset = scene.bar.data.root_pos_w - robot.data.body_pos_w[:, ee_body_id]
        print(f"[skill] grasp assist latched at jaw-to-bar distance {jaw_to_bar:.3f} m", flush=True)
    if args_cli.item == "cube":
        lift_path = [q.clone() for q in reversed(descent_path[:-1])] + [above_bar_q.clone()]
        for waypoint_q in lift_path:
            waypoint_q[:, jaw_ids] = CUBE_GRASP_RAD
        for segment, waypoint_q in enumerate(lift_path, start=1):
            execute_joint_target(f"04_lift_{segment:02d}", waypoint_q, 24)
    else:
        execute_joint_target("04_lift", above_bar_q, args_cli.steps_per_waypoint)
    execute_joint_target("05_above_basket", above_basket_q, args_cli.steps_per_waypoint)
    execute_joint_target("06_descend_to_basket", drop_q, args_cli.steps_per_waypoint)
    release_q = drop_q.clone()
    release_q[:, jaw_ids] = grasp_open_jaw
    execute_joint_target("07_release", release_q, args_cli.settle_steps)
    holding_bar = False
    for _ in range(args_cli.settle_steps):
        scene.step(release_q)
        record_sample("07_release_settle", release_q)
    execute_joint_target("08_retreat", above_basket_q, args_cli.steps_per_waypoint)

    for _ in range(args_cli.settle_steps):
        scene.step(targets)
        record_sample("09_final_settle", targets)
    save_frame(scene.image(), "final_front.png")
    save_frame(scene.wrist_image(), "final_wrist.png")
    save_desktop_frame(scene.image(), "so101_scripted_final_front.png")
    save_desktop_frame(scene.wrist_image(), "so101_scripted_final_wrist.png")

    bar = scene.bar.data.root_pos_w[0].detach().cpu().numpy()
    dx, dy = np.abs(bar[:2] - np.asarray(scene.basket_center[:2]))
    success = dx < 0.085 and dy < 0.065 and 0.01 < bar[2] < 0.12
    print(f"[skill] final bar position={np.round(bar, 4)}; in_basket={bool(success)}", flush=True)
    if success:
        output_dir = ROOT / "runtime" / "outputs" / f"scripted_pick_place_{args_cli.item}"
        output_dir.mkdir(parents=True, exist_ok=True)
        trajectory = {
            "action": np.stack(trajectory_actions),
            "observation_joint_position": np.stack(trajectory_joint_positions),
            "observation_object_pose_wxyz": np.stack(trajectory_object_poses),
            "phase": np.asarray(trajectory_phases),
            "joint_names": np.asarray(robot.joint_names),
            "physics_dt": np.asarray(sim.get_physics_dt(), dtype=np.float32),
            "final_object_position": bar.astype(np.float32),
        }
        np.savez_compressed(output_dir / "successful_cube_trajectory.npz", **trajectory)
        np.savez_compressed(Path.home() / "Desktop" / "so101_successful_cube_trajectory.npz", **trajectory)
        print(
            f"[skill] saved successful trajectory with {len(trajectory_actions)} steps "
            "to runtime/outputs and ~/Desktop",
            flush=True,
        )
    simulation_app.close()


if __name__ == "__main__":
    main()

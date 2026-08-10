#!/usr/bin/env python3
"""Replay the offline real-arm Cartesian plan in Isaac physics.

This validator has no physical robot imports or serial access.  It maps the
30 Hz LeRobot motor targets back into the calibrated Isaac articulation,
executes four 120 Hz physics ticks per target, and records phase diagnostics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--plan", type=Path, default=ROOT / "runtime/outputs/real_cartesian_plan.npz")
parser.add_argument("--save-phase-frames", action="store_true")
parser.add_argument("--grasp-only", action="store_true", help="Stop after the lift, before transport.")
parser.add_argument(
    "--fast-grasp-test",
    action="store_true",
    help="Teleport to the already-validated high pick approach and test only descent/close/lift.",
)
parser.add_argument(
    "--diagnose-close",
    action="store_true",
    help="Stop at the completed close command and save front/wrist diagnostics.",
)
parser.add_argument(
    "--grasp-assist",
    action="store_true",
    help=(
        "Kinematically latch the scanned bar after close, then release it at the "
        "planned release phase. This validates transport/drop geometry when the "
        "lightweight scan collision proxy cannot reproduce real finger friction."
    ),
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

from isaac.bridge import lerobot_to_sim_radians
from isaac.scene import BarPickPlaceScene


OUTPUT_DIR = ROOT / "runtime/outputs/real_cartesian_validation"


def rotate_vector(quat_wxyz: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate batched vectors by batched scalar-first quaternions."""
    quat_vector = quat_wxyz[:, 1:]
    twice_cross = 2.0 * torch.cross(quat_vector, vector, dim=-1)
    return vector + quat_wxyz[:, :1] * twice_cross + torch.cross(
        quat_vector, twice_cross, dim=-1
    )


def save_rgb(image: np.ndarray, filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(image)).save(OUTPUT_DIR / filename)


def main() -> None:
    if not args_cli.plan.exists():
        raise FileNotFoundError(args_cli.plan)
    plan = np.load(args_cli.plan)
    motor_targets = np.asarray(plan["motor_targets"], dtype=np.float32)
    phases = plan["phase"].astype(str)
    if motor_targets.ndim != 2 or motor_targets.shape[1] != 6:
        raise RuntimeError(f"Unexpected plan shape: {motor_targets.shape}")

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=1.0 / 120.0,
            render_interval=4 if args_cli.save_phase_frames else 16,
            device=args_cli.device,
        )
    )
    scene = BarPickPlaceScene(sim, item="bar")
    sim.reset()
    scene.reset()
    scene.set_front_camera_view()
    robot = scene.robot
    robot.write_joint_stiffness_to_sim(
        torch.tensor([[300, 300, 250, 150, 100, 250]], dtype=torch.float32, device=scene.scene.device)
    )
    robot.write_joint_damping_to_sim(
        torch.tensor([[15, 15, 12, 8, 5, 8]], dtype=torch.float32, device=scene.scene.device)
    )
    robot.write_joint_effort_limit_to_sim(
        torch.tensor([[100, 100, 100, 80, 50, 100]], dtype=torch.float32, device=scene.scene.device)
    )
    for _ in range(12):
        scene.step(robot.data.default_joint_pos.clone())
    if args_cli.save_phase_frames:
        save_rgb(scene.image(), "00_initial_front.png")
        save_rgb(scene.wrist_image(), "00_initial_wrist.png")

    start_index = 0
    if args_cli.fast_grasp_test:
        candidates = np.flatnonzero(phases == "pick_cartesian_00")
        if not len(candidates):
            raise RuntimeError("Plan has no pick_cartesian_00 phase")
        # ``phase`` labels the destination of each interpolation segment.  The
        # last sample carrying this label is the actual high approach pose.
        start_index = int(candidates[-1])
        start_target = torch.as_tensor(
            lerobot_to_sim_radians(motor_targets[start_index]),
            dtype=torch.float32,
            device=scene.scene.device,
        ).unsqueeze(0)
        robot.write_joint_state_to_sim(start_target, torch.zeros_like(start_target))
        for _ in range(12):
            scene.step(start_target)
        print(f"[validate] fast grasp test starts at sample {start_index}", flush=True)

    previous_phase = ""
    holding_bar = False
    held_bar_offset_b = torch.zeros((1, 3), dtype=torch.float32, device=scene.scene.device)
    peak_object_height = float(scene.bar.data.root_pos_w[0, 2].item())
    phase_heights: dict[str, float] = {}
    for index in range(start_index, len(motor_targets)):
        motor = motor_targets[index]
        phase = phases[index]
        target = torch.as_tensor(
            lerobot_to_sim_radians(motor), dtype=torch.float32, device=scene.scene.device
        ).unsqueeze(0)
        if args_cli.grasp_assist and phase.startswith("lift_") and not holding_bar:
            holding_bar = True
            gripper_body_ids, _ = robot.find_bodies("gripper")
            if len(gripper_body_ids) != 1:
                raise RuntimeError("Unexpected SO-101 gripper body layout")
            offset_w = (
                scene.bar.data.root_pos_w - robot.data.body_pos_w[:, gripper_body_ids[0]]
            )
            gripper_quat = robot.data.body_quat_w[:, gripper_body_ids[0]]
            inverse_quat = gripper_quat.clone()
            inverse_quat[:, 1:] *= -1.0
            held_bar_offset_b = rotate_vector(inverse_quat, offset_w)
            print("[validate] grasp assist latched after completed close phase", flush=True)
        if args_cli.grasp_assist and phase.startswith("retreat") and holding_bar:
            holding_bar = False
            print("[validate] grasp assist released after completed release phase", flush=True)
        if args_cli.diagnose_close and phase.startswith("lift_"):
            jaw_ids, _ = robot.find_joints("Jaw")
            force_matrix = scene.contact_grasp.data.force_matrix_w
            contact_force = (
                float(torch.linalg.vector_norm(force_matrix, dim=-1).sum().item())
                if force_matrix is not None
                else 0.0
            )
            print(
                f"[validate] completed close: actual jaw={float(robot.data.joint_pos[0, jaw_ids[0]]):.4f} rad "
                f"target jaw={float(target[0, jaw_ids[0]]):.4f} rad "
                f"jaw-filtered contact={contact_force:.3f} N",
                flush=True,
            )
            save_rgb(scene.image(), "diagnose_close_front.png")
            save_rgb(scene.wrist_image(), "diagnose_close_wrist.png")
            save_rgb(scene.image(), str(Path.home() / "Desktop" / "so101_bar_close_front.png"))
            save_rgb(scene.wrist_image(), str(Path.home() / "Desktop" / "so101_bar_close_wrist.png"))
            print("[validate] saved completed-close diagnostics; stopping before lift", flush=True)
            simulation_app.close()
            return
        if args_cli.grasp_only and phase.startswith("transport_"):
            print("[validate] grasp-only stop before transport", flush=True)
            break
        for _ in range(4):
            scene.step(target)
            if holding_bar:
                state = scene.bar.data.root_state_w.clone()
                state[:, :3] = (
                    robot.data.body_pos_w[:, gripper_body_ids[0]]
                    + rotate_vector(
                        robot.data.body_quat_w[:, gripper_body_ids[0]], held_bar_offset_b
                    )
                )
                state[:, 7:] = 0.0
                scene.bar.write_root_pose_to_sim(state[:, :7])
                scene.bar.write_root_velocity_to_sim(state[:, 7:])
        object_height = float(scene.bar.data.root_pos_w[0, 2].item())
        peak_object_height = max(peak_object_height, object_height)
        phase_heights[phase] = max(phase_heights.get(phase, -np.inf), object_height)
        if phase != previous_phase:
            print(
                f"[validate] sample={index:04d} phase={phase:28s} "
                f"object={np.round(scene.bar.data.root_pos_w[0].detach().cpu().numpy(), 4)}",
                flush=True,
            )
            if args_cli.save_phase_frames:
                safe_name = "".join(c if c.isalnum() else "_" for c in phase)
                save_rgb(scene.image(), f"{index:04d}_{safe_name}_front.png")
                save_rgb(scene.wrist_image(), f"{index:04d}_{safe_name}_wrist.png")
            previous_phase = phase

    for _ in range(120):
        scene.step(target)
    final_position = scene.bar.data.root_pos_w[0].detach().cpu().numpy()
    basket = np.asarray(scene.basket_center, dtype=np.float32)
    dx, dy = np.abs(final_position[:2] - basket[:2])
    in_basket = bool(dx < 0.085 and dy < 0.065 and 0.01 < final_position[2] < 0.12)
    grasp_lifted = peak_object_height > 0.07
    if args_cli.save_phase_frames:
        save_rgb(scene.image(), "99_final_front.png")
        save_rgb(scene.wrist_image(), "99_final_wrist.png")
        save_rgb(scene.image(), str(Path.home() / "Desktop" / "so101_real_plan_sim_final_front.png"))
        save_rgb(scene.wrist_image(), str(Path.home() / "Desktop" / "so101_real_plan_sim_final_wrist.png"))
    print(f"[validate] peak object height={peak_object_height:.4f} m; grasp_lifted={grasp_lifted}", flush=True)
    print(
        f"[validate] final object={np.round(final_position, 4)} basket={np.round(basket, 4)} "
        f"in_basket={in_basket}",
        flush=True,
    )
    simulation_app.close()
    if not grasp_lifted or (not args_cli.grasp_only and not in_basket):
        raise RuntimeError("Physical simulation validation failed")


if __name__ == "__main__":
    main()

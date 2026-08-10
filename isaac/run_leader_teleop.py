#!/usr/bin/env python3
"""Drive the Isaac Lab bar scene from a physical SO-101 leader arm."""

from __future__ import annotations

import argparse
import pickle
import socket
import struct
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[1]
LEROBOT_SRC = ROOT / "lerobot" / "src"
for path in (ROOT, LEROBOT_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--host", default="127.0.0.1", help="Leader server host.")
parser.add_argument("--port", type=int, default=5560, help="Leader server port.")
parser.add_argument("--steps", type=int, default=600, help="Control steps to run.")
parser.add_argument("--save_every", type=int, default=60, help="Save every Nth RGB frame.")
parser.add_argument(
    "--record_dir",
    default="",
    help="Optional directory to save a local sim demonstration (frames + actions).",
)
parser.add_argument(
    "--preview_only",
    action="store_true",
    help="Keep the sim frozen at the recorded reset pose and do not connect to the leader server.",
)
parser.add_argument("--align", action="store_true", default=True, help="Align to the leader's first pose.")
parser.add_argument("--no_align", action="store_false", dest="align", help="Skip startup alignment.")
parser.add_argument("--align_duration", type=float, default=2.5, help="Alignment duration in seconds.")
parser.add_argument("--leader_port", default="/dev/ttyACM0", help="Physical leader-arm serial port.")
parser.add_argument("--leader_id", default="so101_leader_arm", help="Leader-arm calibration id.")
parser.add_argument(
    "--leader_python",
    default="/home/teo/so101/.venv/bin/python",
    help="Python executable that runs the leader server.",
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
from isaac.scene import FRONT_CAMERA_EYE_M, FRONT_CAMERA_TARGET_M, BarPickPlaceScene


CONTROL_FPS = 30.0
PHYSICS_SUBSTEPS = 4
SO101_LEADER_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def recv_exact(connection: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = connection.recv(size - len(data))
        if not chunk:
            raise ConnectionError("Leader server disconnected.")
        data.extend(chunk)
    return bytes(data)


def recv_message(connection: socket.socket) -> dict[str, object]:
    size = struct.unpack("!Q", recv_exact(connection, 8))[0]
    return pickle.loads(recv_exact(connection, size))


def send_message(connection: socket.socket, message: dict[str, object]) -> None:
    payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    connection.sendall(struct.pack("!Q", len(payload)) + payload)


class LeaderClient:
    def __init__(self, host: str, port: int) -> None:
        self.connection = socket.create_connection((host, port), timeout=30)
        self.connection.settimeout(30)
        self.request({"command": "ping"})

    def request(self, request: dict[str, object]) -> dict[str, object]:
        send_message(self.connection, request)
        response = recv_message(self.connection)
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "Leader server request failed."))
        return response

    def act(self) -> dict[str, float]:
        return dict(self.request({"command": "act"})["action"])

    def close(self) -> None:
        self.connection.close()


def save_frame(image: np.ndarray, name: str) -> None:
    output_dir = ROOT / "runtime/outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(output_dir / name)


def save_demo(record_dir: Path, frames: list[np.ndarray], leader_actions: list[np.ndarray], sim_targets: list[np.ndarray]) -> None:
    record_dir.mkdir(parents=True, exist_ok=True)
    np.save(record_dir / "leader_actions.npy", np.stack(leader_actions, axis=0))
    np.save(record_dir / "sim_targets.npy", np.stack(sim_targets, axis=0))
    for index, frame in enumerate(frames):
        Image.fromarray(frame).save(record_dir / f"frame_{index:04d}.png")


def leader_action_to_array(action: dict[str, float]) -> np.ndarray:
    return np.asarray([float(action[f"{name}.pos"]) for name in SO101_LEADER_JOINTS], dtype=np.float32)


def main() -> None:
    print("[sim] Creating Isaac Lab simulation context...", flush=True)
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=list(FRONT_CAMERA_EYE_M), target=list(FRONT_CAMERA_TARGET_M))

    print("[sim] Spawning SO-101, bar, basket, and camera...", flush=True)
    scene = BarPickPlaceScene(sim)

    print("[sim] Resetting physics...", flush=True)
    sim.reset()
    scene.reset()
    scene.set_front_camera_view()

    targets = scene.robot.data.default_joint_pos.clone()
    if args_cli.preview_only:
        save_frame(scene.image(), "leader_preview_initial.png")
        print("[sim] Preview-only mode: frozen at the recorded real reset pose.", flush=True)
        try:
            for _ in range(args_cli.steps):
                scene.step(targets)
        finally:
            simulation_app.close()
        return

    print(f"[sim] Connecting to leader server at {args_cli.host}:{args_cli.port}...", flush=True)
    leader = LeaderClient(args_cli.host, args_cli.port)
    leader_action = leader.act()

    if args_cli.align:
        first_pose = lerobot_to_sim_radians(leader_action_to_array(leader_action))
        first_pose = torch.from_numpy(first_pose).to(scene.scene.device).unsqueeze(0)
        print("[sim] Aligning sim arm to the leader's first pose...", flush=True)
        for _ in range(max(1, int(args_cli.align_duration * 120))):
            scene.step(first_pose)

    print("[sim] Starting live teleop.", flush=True)
    record_dir = Path(args_cli.record_dir).expanduser() if args_cli.record_dir else None
    recorded_frames: list[np.ndarray] = []
    recorded_actions: list[np.ndarray] = []
    recorded_targets: list[np.ndarray] = []
    try:
        for step in range(args_cli.steps):
            loop_start = time.perf_counter()
            leader_action = leader.act()
            action_vec = leader_action_to_array(leader_action)
            targets = torch.from_numpy(lerobot_to_sim_radians(action_vec)).to(scene.scene.device).unsqueeze(0)
            for _ in range(PHYSICS_SUBSTEPS):
                scene.step(targets)
            frame = scene.image()
            if record_dir is not None:
                recorded_frames.append(frame.copy())
                recorded_actions.append(action_vec.copy())
                recorded_targets.append(targets[0].detach().cpu().numpy().copy())

            if step % 30 == 0:
                obs = scene.robot.data.joint_pos[0, :6].detach().cpu().numpy()
                print(f"[sim] step={step:04d} leader={np.round(action_vec, 1)} sim={np.round(obs, 3)}", flush=True)
            if step == 0:
                save_frame(frame, "leader_teleop_initial.png")
            elif step % args_cli.save_every == 0:
                save_frame(frame, f"leader_teleop_{step:04d}.png")
            time.sleep(max(0.0, 1.0 / CONTROL_FPS - (time.perf_counter() - loop_start)))
    finally:
        leader.close()
        if record_dir is not None and recorded_frames:
            save_demo(record_dir, recorded_frames, recorded_actions, recorded_targets)
        simulation_app.close()


if __name__ == "__main__":
    main()

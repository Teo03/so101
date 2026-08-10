#!/usr/bin/env python3
"""Evaluate the saved real-world ACT bar policy in the nominal Isaac Lab scene."""

from __future__ import annotations

import argparse
import pickle
import socket
import struct
import sys
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--host", default="127.0.0.1", help="Local ACT policy server host.")
parser.add_argument("--port", type=int, default=5557, help="Local ACT policy server port.")
parser.add_argument("--steps", type=int, default=450, help="30 Hz control steps in this rollout.")
parser.add_argument("--save_every", type=int, default=30, help="Save every Nth simulated RGB frame.")
parser.add_argument(
    "--preview_only",
    action="store_true",
    help="Freeze at the recorded real reset pose for camera/layout calibration; do not connect to the policy.",
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

from isaac.bridge import lerobot_to_sim_radians, sim_radians_to_lerobot
from isaac.scene import FRONT_CAMERA_EYE_M, FRONT_CAMERA_TARGET_M, BarPickPlaceScene


def recv_exact(connection: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = connection.recv(size - len(data))
        if not chunk:
            raise ConnectionError("Policy server disconnected.")
        data.extend(chunk)
    return bytes(data)


def recv_message(connection: socket.socket) -> dict[str, Any]:
    size = struct.unpack("!Q", recv_exact(connection, 8))[0]
    return pickle.loads(recv_exact(connection, size))


def send_message(connection: socket.socket, message: dict[str, Any]) -> None:
    payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    connection.sendall(struct.pack("!Q", len(payload)) + payload)


class PolicyClient:
    def __init__(self, host: str, port: int) -> None:
        self.connection = socket.create_connection((host, port), timeout=30)
        self.connection.settimeout(60)
        self.request({"command": "ping"})

    def request(self, request: dict[str, Any]) -> dict[str, Any]:
        send_message(self.connection, request)
        response = recv_message(self.connection)
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "Policy server request failed."))
        return response

    def reset(self) -> None:
        self.request({"command": "reset"})

    def act(self, state: np.ndarray, images: dict[str, np.ndarray]) -> np.ndarray:
        return np.asarray(self.request({"command": "act", "state": state, "images": images})["action"], dtype=np.float32)

    def close(self) -> None:
        self.connection.close()


def save_frame(image: np.ndarray, name: str) -> None:
    output_dir = ROOT / "runtime/outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(output_dir / name)


def main() -> None:
    print("[sim] Creating Isaac Lab simulation context...", flush=True)
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Keep the visible Isaac viewport and the policy observation camera aligned.
    sim.set_camera_view(eye=list(FRONT_CAMERA_EYE_M), target=list(FRONT_CAMERA_TARGET_M))
    print("[sim] Spawning SO-101, bar, basket, and camera...", flush=True)
    scene = BarPickPlaceScene(sim)

    print("[sim] Resetting physics...", flush=True)
    sim.reset()
    scene.reset()
    scene.set_front_camera_view()

    # Allow RTX sensors and physics to settle before the first policy observation.
    targets = scene.robot.data.default_joint_pos.clone()
    for _ in range(8):
        scene.step(targets)

    if args_cli.preview_only:
        save_frame(scene.image(), "initial_frame.png")
        save_frame(scene.wrist_image(), "initial_wrist_frame.png")
        print("[sim] Preview-only mode: frozen at the recorded real reset pose.", flush=True)
        try:
            for _ in range(args_cli.steps):
                scene.step(targets)
        finally:
            simulation_app.close()
        return

    print("[sim] Connecting to policy server...", flush=True)
    client = PolicyClient(args_cli.host, args_cli.port)
    client.reset()
    print("[sim] Connected to ACT policy server; starting zero-shot rollout.")
    try:
        for step in range(args_cli.steps):
            images = scene.images()
            image = images["front"]
            if step == 0:
                save_frame(image, "initial_frame.png")
                save_frame(images["wrist"], "initial_wrist_frame.png")
            elif step % args_cli.save_every == 0:
                save_frame(image, f"frame_{step:04d}.png")
                save_frame(images["wrist"], f"wrist_frame_{step:04d}.png")

            joint_radians = scene.robot.data.joint_pos[0, :6].detach().cpu().numpy()
            real_state = sim_radians_to_lerobot(joint_radians)
            real_action = client.act(real_state, images)
            targets = torch.from_numpy(lerobot_to_sim_radians(real_action)).to(scene.scene.device).unsqueeze(0)

            # Isaac Lab scene uses 120 Hz physics and we run the policy at 30 Hz.
            for _ in range(4):
                scene.step(targets)

            if step % 30 == 0:
                print(
                    f"[sim] step={step:03d} state={np.round(real_state, 1)} "
                    f"action={np.round(real_action, 1)}"
                )
    finally:
        client.close()
        simulation_app.close()


if __name__ == "__main__":
    main()

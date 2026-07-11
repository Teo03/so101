#!/usr/bin/env python3
"""Local inference service for the real-world ACT checkpoint.

Run this with /home/teo/so101/.venv/bin/python.  It deliberately uses the same
LeRobot environment that trained the checkpoint, avoiding package conflicts with
Isaac Lab's Python environment.
"""

from __future__ import annotations

import argparse
import pickle
import socket
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.common.control_utils import predict_action
from lerobot.policies.act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "lerobot/outputs/train/act_bar_pickplace_40ep/checkpoints/020000/pretrained_model"


def recv_exact(connection: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = connection.recv(size - len(data))
        if not chunk:
            raise ConnectionError("Simulation client disconnected.")
        data.extend(chunk)
    return bytes(data)


def recv_message(connection: socket.socket) -> dict[str, Any]:
    size = struct.unpack("!Q", recv_exact(connection, 8))[0]
    return pickle.loads(recv_exact(connection, size))


def send_message(connection: socket.socket, message: dict[str, Any]) -> None:
    payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    connection.sendall(struct.pack("!Q", len(payload)) + payload)


class ACTInferenceServer:
    def __init__(self, checkpoint: Path, device: str) -> None:
        self.device = torch.device(device)
        self.policy = ACTPolicy.from_pretrained(checkpoint).to(self.device)
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )
        self.policy.eval()
        print(f"[policy] Loaded {checkpoint} on {self.device}.", flush=True)

    def reset(self) -> None:
        self.policy.reset()

    def act(self, state: np.ndarray, image: np.ndarray) -> np.ndarray:
        if state.shape != (6,):
            raise ValueError(f"Expected state shape (6,), got {state.shape}.")
        if image.shape != (480, 640, 3):
            raise ValueError(f"Expected front image shape (480, 640, 3), got {image.shape}.")
        action = predict_action(
            observation={
                "observation.state": state.astype(np.float32, copy=False),
                "observation.images.front": image.astype(np.uint8, copy=False),
            },
            policy=self.policy,
            device=self.device,
            preprocessor=self.preprocessor,
            postprocessor=self.postprocessor,
            use_amp=self.policy.config.use_amp,
            task="Pick up the bar and put it in the basket",
            robot_type="so_follower",
        )
        return action.squeeze(0).detach().cpu().numpy().astype(np.float32)


def serve(server: ACTInferenceServer, host: str, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, port))
        listener.listen(1)
        print(f"[policy] Listening on {host}:{port}.", flush=True)
        while True:
            connection, address = listener.accept()
            print(f"[policy] Simulation client connected from {address}.", flush=True)
            try:
                with connection:
                    while True:
                        request = recv_message(connection)
                        command = request.get("command")
                        if command == "reset":
                            server.reset()
                            send_message(connection, {"ok": True})
                        elif command == "act":
                            action = server.act(request["state"], request["image"])
                            # A Python list avoids cross-environment NumPy pickle compatibility
                            # issues (LeRobot uses Python 3.12; Isaac Lab uses Python 3.11).
                            send_message(connection, {"ok": True, "action": action.tolist()})
                        elif command == "ping":
                            send_message(connection, {"ok": True})
                        else:
                            raise ValueError(f"Unknown command: {command!r}")
            except (ConnectionError, OSError) as error:
                print(f"[policy] Client disconnected: {error}", flush=True)
            except Exception as error:  # keep the server alive for the next simulation launch
                print(f"[policy] Request failed: {error}", flush=True)
                try:
                    send_message(connection, {"ok": False, "error": str(error)})
                except OSError:
                    pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5557)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    serve(ACTInferenceServer(args.checkpoint, args.device), args.host, args.port)


if __name__ == "__main__":
    main()

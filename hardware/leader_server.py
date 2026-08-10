#!/usr/bin/env python3
"""Expose a physical SO-101 leader arm over a small TCP bridge."""

from __future__ import annotations

import argparse
import pickle
import socket
import struct
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEROBOT_SRC = ROOT / "lerobot" / "src"
for path in (ROOT, LEROBOT_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lerobot.teleoperators.so_leader import SO101Leader, SOLeaderTeleopConfig


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--host", default="127.0.0.1", help="TCP bind host.")
parser.add_argument("--port", type=int, default=5560, help="TCP bind port.")
parser.add_argument("--leader_port", default="/dev/ttyACM0", help="Physical leader-arm serial port.")
parser.add_argument("--leader_id", default="so101_leader_arm", help="Leader-arm calibration id.")
args = parser.parse_args()


def recv_exact(connection: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = connection.recv(size - len(data))
        if not chunk:
            raise ConnectionError("Client disconnected.")
        data.extend(chunk)
    return bytes(data)


def recv_message(connection: socket.socket) -> dict[str, object]:
    size = struct.unpack("!Q", recv_exact(connection, 8))[0]
    return pickle.loads(recv_exact(connection, size))


def send_message(connection: socket.socket, message: dict[str, object]) -> None:
    payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    connection.sendall(struct.pack("!Q", len(payload)) + payload)


leader = SO101Leader(SOLeaderTeleopConfig(port=args.leader_port, id=args.leader_id, use_degrees=True))
print(f"[leader] Connecting to {args.leader_port}...", flush=True)
leader.connect()
print("[leader] Connected.", flush=True)

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((args.host, args.port))
server.listen(1)
print(f"[leader] Serving on {args.host}:{args.port}", flush=True)

try:
    while True:
        connection, _ = server.accept()
        with connection:
            connection.settimeout(30)
            while True:
                try:
                    request = recv_message(connection)
                except ConnectionError:
                    break
                except Exception as exc:
                    send_message(connection, {"ok": False, "error": str(exc)})
                    break

                command = request.get("command")
                if command == "ping":
                    send_message(connection, {"ok": True})
                elif command == "act":
                    try:
                        send_message(connection, {"ok": True, "action": leader.get_action()})
                    except Exception as exc:
                        send_message(connection, {"ok": False, "error": str(exc)})
                else:
                    send_message(connection, {"ok": False, "error": f"Unknown command: {command!r}"})
finally:
    leader.disconnect()
    server.close()

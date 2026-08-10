#!/usr/bin/env python3
"""Canonical command index for the SO-101 workspace.

This dispatcher intentionally executes the existing scripts instead of
importing them. Isaac scripts construct SimulationApp during import, while real
hardware scripts need their existing process/signal boundaries.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ISAACLAB = Path("/home/teo/IsaacLab/isaaclab.sh")


@dataclass(frozen=True)
class Command:
    group: str
    name: str
    script: str
    description: str
    isaac: bool = False


COMMANDS = (
    Command("cap", "observe", "perception/perceive_scene.py", "Detect the bar and basket."),
    Command("cap", "plan", "code_as_policy/plan_from_scene.py", "Write an inspectable task/skill program."),
    Command("cap", "run", "code_as_policy/run_vision_pick_place.py", "Run the bounded vision task pipeline."),
    Command("real", "plan", "hardware/plan_pick_place.py", "Compile standalone Cartesian IK."),
    Command("real", "execute", "hardware/run_real_cartesian.py", "Inspect or execute a guarded motor plan."),
    Command("real", "sync", "hardware/sync_leader_to_follower.py", "Align leader and follower explicitly."),
    Command("real", "watchdog", "hardware/real_torque_watchdog.py", "Run the follower torque watchdog."),
    Command("service", "act", "act/policy_server.py", "Serve a saved ACT policy locally."),
    Command("service", "leader", "hardware/leader_server.py", "Serve physical leader observations."),
    Command("sim", "act", "isaac/run_zero_shot.py", "Evaluate ACT in the calibrated scene.", True),
    Command("sim", "teleop", "isaac/run_leader_teleop.py", "Teleoperate the calibrated scene.", True),
    Command("sim", "scripted", "isaac/run_scripted_pick_place.py", "Run the scripted Isaac skill.", True),
    Command("sim", "validate", "isaac/validate_real_cartesian_plan.py", "Replay a motor plan in physics.", True),
    Command("rl", "check", "rl/check_rl_env.py", "Check environment resets and stepping.", True),
    Command("rl", "train", "rl/train_rl.py", "Train the experimental PPO teacher.", True),
    Command("rl", "play", "rl/play_rl.py", "Render the newest PPO checkpoint.", True),
    Command("asset", "convert", "assets/convert_scans_to_usd.py", "Convert scanned GLBs to USD.", True),
)


def print_commands() -> None:
    print("Usage: python -m tools.cli <group> <command> [arguments...]\n")
    width = max(len(f"{item.group} {item.name}") for item in COMMANDS)
    for item in COMMANDS:
        label = f"{item.group} {item.name}"
        runtime = "Isaac" if item.isaac else "Python"
        print(f"  {label:<{width}}  [{runtime:6s}] {item.description}")
    print("\nExamples:")
    print("  python -m tools.cli cap observe --device 0")
    print("  python -m tools.cli cap plan")
    print("  python -m tools.cli cap run")
    print("  python -m tools.cli real execute --plan PLAN.npz")
    print("  python -m tools.cli sim act --preview_only --steps 1 --device cuda")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"list", "-h", "--help"}:
        print_commands()
        return
    if len(sys.argv) < 3:
        raise SystemExit("Expected <group> <command>; use 'list' to see available commands")
    group, name = sys.argv[1:3]
    selected = next(
        (item for item in COMMANDS if item.group == group and item.name == name),
        None,
    )
    if selected is None:
        raise SystemExit(f"Unknown command {group!r} {name!r}; use 'list'")

    script = ROOT / selected.script
    arguments = sys.argv[3:]
    if selected.isaac:
        if not ISAACLAB.exists():
            raise SystemExit(f"Isaac Lab launcher not found: {ISAACLAB}")
        env = os.environ.copy()
        env.setdefault("TERM", "xterm-256color")
        env.setdefault("DISPLAY", ":0")
        env.setdefault("XAUTHORITY", "/run/user/1000/gdm/Xauthority")
        env.setdefault("CONDA_PREFIX", "/home/teo/miniconda3/envs/env_isaaclab")
        os.execve(str(ISAACLAB), [str(ISAACLAB), "-p", str(script), *arguments], env)
    os.execv(sys.executable, [sys.executable, str(script), *arguments])


if __name__ == "__main__":
    main()

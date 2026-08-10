#!/usr/bin/env python3
"""Open one RL environment and verify reward/reset signals before training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=180)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from rl.rl_env import SO101BarBucketEnv, SO101BarBucketEnvCfg


def main() -> None:
    cfg = SO101BarBucketEnvCfg()
    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = 2.0
    cfg.sim.device = args_cli.device or "cuda:0"
    env = SO101BarBucketEnv(cfg)
    obs, _ = env.reset()
    print(f"[rl] observation shape: {tuple(obs['policy'].shape)}", flush=True)
    for step in range(args_cli.steps):
        actions = 0.15 * torch.randn((1, cfg.action_space), device=env.device)
        _, reward, terminated, truncated, _ = env.step(actions)
        if step % 30 == 0:
            print(f"[rl] step={step:03d} reward={reward.item():.3f} terminated={terminated.item()} timeout={truncated.item()}", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

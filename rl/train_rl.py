#!/usr/bin/env python3
"""Train the privileged-state PPO teacher for bar-to-bucket pick-and-place."""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=64, help="Parallel environments; start at 16 on this PC.")
parser.add_argument("--max_iterations", type=int, default=1500)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from rl.rl_agent_cfg import SO101BarBucketPPORunnerCfg
from rl.rl_env import SO101BarBucketEnv, SO101BarBucketEnvCfg


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    cfg = SO101BarBucketEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device or "cuda:0"
    cfg.seed = args_cli.seed
    agent_cfg = SO101BarBucketPPORunnerCfg()
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.seed = args_cli.seed
    log_dir = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make("SO101-BarBucket-v0", cfg=cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=cfg.sim.device)
    print(f"[rl] Training {cfg.scene.num_envs} environments; logs: {log_dir}", flush=True)
    started = time.time()
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    print(f"[rl] Training complete in {time.time() - started:.1f}s", flush=True)
    env.close()


if __name__ == "__main__":
    gym.register("SO101-BarBucket-v0", entry_point=SO101BarBucketEnv, disable_env_checker=True)
    try:
        main()
    finally:
        simulation_app.close()

#!/usr/bin/env python3
"""Watch a trained state-based PPO bar-to-basket policy in Isaac Sim."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = ROOT / "logs/rsl_rl/so101_bar_bucket_ppo"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, default=None, help="Explicit model_*.pt path; defaults to the newest PPO checkpoint.")
parser.add_argument("--steps", type=int, default=10_000, help="Maximum visible policy steps.")
parser.add_argument("--playback_hz", type=float, default=5.0, help="Visible playback rate; use 5 Hz for inspection.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from rl.rl_agent_cfg import SO101BarBucketPPORunnerCfg
from rl.rl_env import SO101BarBucketEnv, SO101BarBucketEnvCfg


def latest_checkpoint() -> Path:
    candidates = sorted(DEFAULT_RUN_DIR.glob("*/model_*.pt"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No PPO checkpoint found below {DEFAULT_RUN_DIR}")
    return candidates[-1]


def main() -> None:
    checkpoint = args_cli.checkpoint or latest_checkpoint()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    cfg = SO101BarBucketEnvCfg()
    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = 2.0
    cfg.sim.device = args_cli.device or "cuda:0"
    agent_cfg = SO101BarBucketPPORunnerCfg()

    env = gym.make("SO101-BarBucket-v0", cfg=cfg, render_mode=None)
    vec_env = RslRlVecEnvWrapper(env, clip_actions=1.0)
    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=str(checkpoint.parent), device=cfg.sim.device)
    runner.load(str(checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=cfg.sim.device)

    # Match the calibrated overview used for the policy camera comparison.
    vec_env.unwrapped.sim.set_camera_view(eye=[-0.02, -0.55, 0.50], target=[0.03, 0.08, 0.04])
    policy_nn = runner.alg.policy
    obs = vec_env.get_observations()
    print(f"[rl] Loaded {checkpoint}; showing policy rollout on DISPLAY={args_cli.display if hasattr(args_cli, 'display') else ':0'}", flush=True)
    try:
        # Follow Isaac Lab's official RSL-RL player lifecycle: keep stepping
        # until the visible app is closed.  The fixed step cap is only a safety
        # limit, not the normal reason the player terminates.
        timestep = 0
        while simulation_app.is_running() and timestep < args_cli.steps:
            start_time = time.time()
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = vec_env.step(actions)
                policy_nn.reset(dones)
            timestep += 1
            sleep_time = (1.0 / args_cli.playback_hz) - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        vec_env.close()


if __name__ == "__main__":
    gym.register("SO101-BarBucket-v0", entry_point=SO101BarBucketEnv, disable_env_checker=True)
    try:
        main()
    finally:
        simulation_app.close()

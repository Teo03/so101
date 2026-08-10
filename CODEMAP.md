# SO-101 code map

The project is organized into isolated top-level packages. `python -m tools.cli`
is the canonical user-facing command index.

## Code as Policy

- `code_as_policy/run_vision_pick_place.py`: bounded observe → task program → IK → real execution
  pipeline.
- `code_as_policy/plan_from_scene.py`: converts grounded objects into the inspectable
  `so101-task-program-v1` skill program.
- Runtime policy: `runtime/outputs/vision_task_plan.json` (generated and ignored).

This is currently deterministic program generation. An LLM planner can later
target the same task-program schema without receiving serial or raw motor access.

## Perception and workspace geometry

- `perception/perceive_scene.py`: prompted YOLOE masks for bar and basket.
- `perception/workspace_geometry.py`: calibrated pixel-to-base homography and yaw mapping.
- `perception/calibrate_workspace.py`: general calibration fitting.
- `perception/calibrate_current_workspace.py`: current fixed-workspace calibration.
- `perception/detect_real_bar.py`: older task-specific perception experiment.

## Real hardware

- `hardware/run_real_cartesian.py`: guarded follower executor, telemetry trips, retrace,
  watchdog integration, and torque teardown.
- `hardware/real_torque_watchdog.py`: detached torque fail-safe.
- `hardware/sync_leader_to_follower.py`: explicit leader/follower alignment utility.
- `hardware/test_leader_actuation.py`: bounded leader hardware diagnostic.

## Learned ACT policies and teleoperation

- `act/policy_server.py`: loads the saved ACT checkpoint behind a local API.
- `hardware/leader_server.py`: exposes the physical leader to Isaac teleoperation.
- `isaac/run_zero_shot.py`: evaluates the real ACT policy in the calibrated sim.
- `isaac/run_leader_teleop.py`: drives Isaac from the leader and optionally records.

ACT is a motor skill. Code as Policy can select or supervise it; it does not
replace ACT's 30 Hz visual-motor control.

## Isaac scene and Cartesian skills

- `isaac/scene.py`: robot, scanned objects, workspace, contacts, and cameras.
- `isaac/bridge.py`: LeRobot motor-space ↔ Isaac-radian conversion.
- `isaac/plan_real_cartesian.py`: joint-limited IK compiler for real execution.
- `isaac/validate_real_cartesian_plan.py`: physics replay/trajectory validator.
- `isaac/run_scripted_pick_place.py`: deterministic Isaac pick-and-place development
  skill.
- `assets/convert_scans_to_usd.py`: GLB → textured USD conversion.
- `assets/scans/`: GLB/USD assets and their relative texture tree.

## Reinforcement learning experiment

- `rl/rl_env.py`: state-based Isaac Lab teacher environment.
- `rl/rl_agent_cfg.py`: PPO configuration.
- `rl/check_rl_env.py`, `rl/train_rl.py`, `rl/play_rl.py`: entry points.

RL remains an experimental data-generation route. Its checkpoints are not
authorized for direct real-robot deployment.

## Runtime data

Generated files belong under `runtime/outputs/` or `runtime/demos/` and are
ignored by Git. Large local perception models belong under `models/`.
The `lerobot/outputs/` tree belongs to the LeRobot training workflow.

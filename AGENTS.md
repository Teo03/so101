# AGENTS.md

## Purpose

This repository captures the SO-101 sim-to-real bar pick-and-place calibration work.

## What was done

- Recreated the real SO-101 workspace in Isaac Sim / Isaac Lab with the bar on the left, basket on the right, and the robot posed to match the recorded demonstrations.
- Tuned the policy camera so the sim view matches the real Logitech C270 framing more closely.
- Added a leader-arm teleoperation bridge for `/dev/ttyACM0` so the physical SO-101 leader can drive the Isaac scene live.
- Verified the saved real ACT checkpoint in sim and confirmed the rollout is coherent.
- Recorded a short sim demonstration bundle for later dataset conversion and training.

## Important files

- `sim_bar/scene.py`: Isaac scene, workspace, and camera calibration.
- `sim_bar/run_zero_shot.py`: real-policy evaluation in sim.
- `sim_bar/run_leader_teleop.py`: live leader-arm teleoperation in sim.
- `sim_bar/leader_server.py`: TCP bridge that reads the physical leader arm.
- `sim_bar/policy_server.py`: TCP bridge for the saved ACT checkpoint.
- `sim_bar/README.md`: working run instructions.

## Current workflow

1. Start `sim_bar/leader_server.py` with the physical leader on `/dev/ttyACM0`.
2. Run `sim_bar/run_leader_teleop.py` in Isaac Lab on `DISPLAY=:0`.
3. Use `--record_dir` to save sim demos.
4. Run `sim_bar/policy_server.py` and `sim_bar/run_zero_shot.py` to test the real model in sim.

## Notes

- The repository intentionally keeps the cloned `lerobot/` checkout out of version control.
- Generated sim demos and render outputs are not meant to be committed by default.
- Consolidated operating notes from the earlier project memory:
  - Workspace: `/home/teo/so101`
  - Hardware variant: 5V/7.4V SO-101
  - Leader arm on `/dev/ttyACM0`; follower arm was used on `/dev/ttyACM1` in earlier runs, but verify by motion if the USB order changes
  - The follower arm is for rollout; the leader arm is for teleoperation and data collection
  - Reliable camera mode was `640x480` at `30 FPS`; `1280x720` YUYV was too slow
  - Keep camera users closed during recording; browser preview can hold the webcam
  - For recording, `Right`/`n` advances, `Left`/`r` re-records, `Esc`/`q` stops
  - Force-killed recording can leave torque state uncertain; power-cycle the follower if needed
  - ACT training succeeded on the 30-episode red-ball task and the 40-episode bar task
  - Key checkpoints:
    - `/home/teo/so101/lerobot/outputs/train/act_pickplace_30ep/checkpoints/last/pretrained_model`
    - `/home/teo/so101/lerobot/outputs/train/act_bar_pickplace_40ep/checkpoints/last/pretrained_model`

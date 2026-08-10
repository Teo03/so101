# AGENTS.md

## Purpose

This repository captures the SO-101 sim-to-real bar pick-and-place calibration work.

## What was done

- Recreated the real SO-101 workspace in Isaac Sim / Isaac Lab with the bar on the left, basket on the right, and the robot posed to match the recorded demonstrations.
- Tuned the policy camera so the sim view matches the real Logitech C270 framing more closely.
- Added a leader-arm teleoperation bridge for `/dev/ttyACM0` so the physical SO-101 leader can drive the Isaac scene live.
- Verified the saved real ACT checkpoint in sim and confirmed the rollout is coherent.
- Recorded a short sim demonstration bundle for later dataset conversion and training.
- Replaced prototype placeholders with scanned assets for the basket and chocolate bar, then converted the textured GLB scans into USD for Isaac Sim.
- Added a lightweight collision-proxy approach for scanned assets so the visible object can stay visually accurate while still supporting teleop and policy evaluation.
- Matched the policy camera, teleop camera, and visible viewport so the same framing is used for sim previews, rollouts, and later real/sim comparisons.
- Set up the sim workspace to mirror the real desk: bar to the left of the arm, basket to the right, warm wood tabletop, three-sided white enclosure, and the robot reset to the same right-arm pose used in the demonstrations.

## Important files

- `sim_bar/scene.py`: Isaac scene, workspace, and camera calibration.
- `sim_bar/run_zero_shot.py`: real-policy evaluation in sim.
- `sim_bar/run_leader_teleop.py`: live leader-arm teleoperation in sim.
- `sim_bar/leader_server.py`: TCP bridge that reads the physical leader arm.
- `sim_bar/policy_server.py`: TCP bridge for the saved ACT checkpoint.
- `sim_bar/convert_scans_to_usd.py`: converts textured scan assets into Isaac-friendly USD.
- `sim_bar/README.md`: working run instructions.
- `lerobot/`: private LeRobot submodule containing the validated real-robot DAgger changes.

## Current workflow

1. Start `sim_bar/leader_server.py` with the physical leader on `/dev/ttyACM0`.
2. Run `sim_bar/run_leader_teleop.py` in Isaac Lab on `DISPLAY=:0`.
3. Use `--record_dir` to save sim demos when collecting data.
4. Compare the frozen preview frame from `sim_bar/run_zero_shot.py --preview_only` against the latest real camera shot before changing object placement or scale.
5. Run `sim_bar/policy_server.py` and `sim_bar/run_zero_shot.py` to test the real model in sim.
6. For new workspaces, prefer scanning the real objects, converting them to USD, importing the scanned assets into Isaac, and then validating with the same camera before collecting demonstrations or training.

## Validated real-robot DAgger correction workflow

The ACT correction workflow was physically validated on 2026-08-09. The customized
LeRobot source is preserved in the private repository
`https://github.com/Teo03/lerobot-so101-dagger` and linked here as the `lerobot/`
submodule. The validated private commit is `c21eac60` (`feat(rollout): add robust
DAgger corrections`). The corresponding development commit with the upstream LeRobot
history is retained locally as `6baf10b0` on `agent/robust-dagger-corrections`.

The correction flow uses one key:

1. ACT is running autonomously. Press `Space` once to pause the follower. The measured
   follower pose is captured and held, and the leader automatically moves to that exact
   pose.
2. After alignment completes, press `Space` again. Leader torque is released and
   correction recording begins; direct leader-to-follower mapping is the same absolute
   joint mapping used by `lerobot-teleoperate`.
3. When the correction is complete, press `Space` a third time. The correction episode
   is saved, the leader automatically returns to its startup pose, the ACT inference
   state is reset, and autonomous execution resumes.
4. Repeat the three-state cycle for each correction. `Tab` is not needed.
5. Press `Esc` at any point to stop. With `--return_to_initial_position=true`, teardown
   stops inference, returns the follower to the pose captured at startup, disconnects
   both cameras and both arms, finalizes the dataset, and performs a bounded Rerun
   shutdown. `Ctrl-C` follows the same teardown path.

Why the mapping now works:

- `--strategy.relative_clutch_handover=false` selects direct absolute leader/follower
  mapping, matching normal `lerobot-teleoperate` behavior.
- The automatic handover targets the measured follower joint pose, not the last raw ACT
  target. This prevents a discontinuity between autonomous control and correction.
- Leader torque enable retries each servo and rolls back already-enabled servos if a
  later servo fails. Smooth-move errors also disable torque best-effort.
- Do not set a global `--robot.max_relative_target=5` for this workflow. It clamps ACT
  autonomous actions and previously caused a continuous clamp-warning storm and badly
  distorted motion.
- Keep `--policy.n_action_steps=100`; this is the physically validated setting for this
  checkpoint.

Validated command:

```bash
DISPLAY=:0 \
XAUTHORITY=/run/user/1000/gdm/Xauthority \
LEROBOT_FORCE_TERMINAL_KEYBOARD=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128 \
./.venv/bin/lerobot-rollout \
  --strategy.type=dagger \
  --strategy.relative_clutch_handover=false \
  --strategy.smooth_leader_to_follower_handover=true \
  --strategy.record_autonomous=false \
  --strategy.num_episodes=10 \
  --strategy.input_device=keyboard \
  --inference.type=sync \
  --policy.path=/home/teo/so101/lerobot/outputs/train/act_hotwheels_hanging_35ep/checkpoints/last/pretrained_model \
  --policy.n_action_steps=100 \
  --robot.type=so101_follower \
  --robot.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D049262-if00 \
  --robot.id=my_follower \
  --robot.cameras='{"front":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30,"warmup_s":3,"fourcc":"MJPG"},"wrist":{"type":"opencv","index_or_path":2,"width":640,"height":360,"fps":31,"warmup_s":5,"fourcc":"YUYV"}}' \
  --teleop.type=so101_leader \
  --teleop.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14030274-if00 \
  --teleop.id=my_leader \
  --dataset.repo_id=local/rollout_hotwheels_corrections_v6 \
  --dataset.single_task="Pick up the packaged Hot Wheels car and hang it on the empty hook" \
  --dataset.fps=30 \
  --dataset.num_episodes=10 \
  --dataset.push_to_hub=false \
  --dataset.streaming_encoding=false \
  --fps=30 \
  --duration=3000 \
  --display_data=true \
  --display_mode=rerun \
  --display_compressed_images=true \
  --return_to_initial_position=true
```

Operational notes:

- The wrist camera reports `31 FPS`; configure it as `fps=31`. Asking for `30` fails
  LeRobot's exact OpenCV FPS validation because the camera reports `actual_fps=31.0`.
- Keep the leader's long handle and the workspace clear during automatic alignment and
  automatic return-to-start.
- Initial inference, the intentional smooth handover, and AV1 episode encoding can emit
  temporary slow-loop warnings. Compressed Rerun images and bounded Rerun shutdown avoid
  the previous visualization backpressure hang.
- If a run is force-killed or motor torque state is uncertain, power-cycle the affected
  arm before touching it or starting another run.
- Corrections collected before exact mapping was fixed (notably early `v1`/`v2` runs)
  should not be mixed into training. Use a fresh dataset repo ID for each new collection.
- Software validation for the private commit: `34 passed` in `tests/test_rollout.py`,
  changed-file Ruff checks passed, `git diff --check` passed, and Python compilation
  passed. The complete state cycle was then validated on the physical leader/follower.

## Notes

- `lerobot/` is a private git submodule pinned to the validated source snapshot. After a
  fresh clone, initialize it with `git submodule update --init --recursive`; GitHub access
  to `Teo03/lerobot-so101-dagger` is required.
- Generated sim demos and render outputs are not meant to be committed by default.
- For scanned assets, preserve the textured visible mesh, then add a simple hidden collision proxy if the asset does not already include usable physics.
- If the sim view drifts from the real view, first compare `sim_bar/outputs/initial_frame.png` with the real camera frame before changing the robot or policy.
- Tricky but useful commands:
  - `TERM=xterm-256color DISPLAY=:0 CONDA_PREFIX=/home/teo/miniconda3/envs/env_isaaclab /home/teo/IsaacLab/isaaclab.sh -p /home/teo/so101/sim_bar/run_zero_shot.py --preview_only --steps 1 --rendering_mode balanced --device cuda`
  - `TERM=xterm-256color DISPLAY=:0 CONDA_PREFIX=/home/teo/miniconda3/envs/env_isaaclab /home/teo/IsaacLab/isaaclab.sh -p /home/teo/so101/sim_bar/run_leader_teleop.py --host 127.0.0.1 --port 5560 --steps 100000 --save_every 300 --rendering_mode balanced --device cuda`
  - `./.venv/bin/python sim_bar/leader_server.py --leader_port /dev/ttyACM0 --leader_id my_leader --port 5560`
  - `./.venv/bin/python sim_bar/policy_server.py --port 5557`
- The policy and teleop scripts save their first camera frame automatically, so the fast calibration loop is: run preview, compare `sim_bar/outputs/initial_frame.png`, adjust `sim_bar/scene.py`, rerun.
- Scan conversion settings that mattered:
  - Use Isaac Sim's asset converter in a headless `SimulationApp`.
  - Enable `embed_textures=True` and `export_preview_surface=True` so the scan stays visually faithful inside USD.
  - Set `use_meter_as_world_unit=True` to keep the imported scan sizes consistent with Isaac Lab.
  - Ignore animations, cameras, and lights when converting scan meshes.
- The object setup flow is now: scan the real object, convert to textured USD, import the USD into `sim_bar/scene.py`, tune scale/orientation/placement against the same camera view, then validate with teleop or the saved policy before collecting more demos.
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
- Waddle-style non-ACT proof completed on 2026-07-30:
  - `sim_bar/perceive_scene.py` uses prompted YOLOE segmentation for the bar and basket.
  - `sim_bar/workspace_geometry.py` maps pixels to robot-base XY through a planar homography.
  - `sim_bar/plan_from_scene.py` writes a reusable task/skill program and Isaac Lab IK command.
  - The real bar was successfully localized, grasped, lifted, lowered, and released using
    detected X/Y plus the physically proven wrist orientation.
  - A complete second run transported the bar and visibly dropped it inside the basket;
    verified frame: `~/Desktop/so101_success_bar_in_basket.png`.
  - The explicit open-gripper `postdrop_*` path returns home without re-closing in the
    basket. Treat these phases with the validated return-load telemetry envelope; the
    tighter forward threshold caused an unnecessary fallback retrace after the first drop.
  - Evidence: `~/Desktop/so101_vision_pick_lift/lift_cartesian_00_{front,wrist}.png`.
  - Vision-derived wrist yaw is not yet physically validated. Keep `orientation-mode=proven`
    unless a staged wrist-camera check confirms both fingers straddle the object.
  - For a new camera/table setup, collect at least four spread-out pixel/robot XY
    correspondences and use `sim_bar/calibrate_workspace.py`; do not reuse the current
    homography after moving the camera.

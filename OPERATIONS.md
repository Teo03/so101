# SO-101 bar pick-and-place: zero-shot simulation evaluation

> This is the detailed operations guide. For subsystem ownership and the
> relationship between Code as Policy, ACT, real control, Isaac, and RL, see
> [`CODEMAP.md`](CODEMAP.md). New commands can be discovered with
> `./.venv/bin/python -m tools.cli list`; the original script paths below
> remain supported for reproducibility.

This project evaluates the existing real-world ACT checkpoint in Isaac Lab before
collecting simulated demonstrations.  It intentionally keeps the trained policy in
the existing LeRobot virtual environment and runs Isaac Lab in its existing Conda
environment.  A local TCP bridge passes one `640x480` RGB frame and six SO-101
joint values to the policy server.

## What is matched to the real policy

- camera key: `observation.images.front`
- RGB resolution: `640x480`
- state/action order: shoulder pan, shoulder lift, elbow flex, wrist flex, wrist
  roll, gripper
- action/state units: LeRobot SO-101 normalized motor-space values, with the
  workshop's calibrated conversion to the Isaac USD joint ranges
- policy: `act_bar_pickplace_40ep`, checkpoint `020000`

## Run

Open two terminals.

```bash
# Terminal 1: expose the physical leader arm over TCP.
cd /home/teo/so101
./.venv/bin/python hardware/leader_server.py --leader_port /dev/ttyACM0 --leader_id my_leader
```

```bash
# Terminal 2: launch Isaac Sim/Isaac Lab with the nominal bar scene.
cd /home/teo/IsaacLab
TERM=xterm-256color DISPLAY=:0 CONDA_PREFIX=/home/teo/miniconda3/envs/env_isaaclab \
  ./isaaclab.sh -p /home/teo/so101/isaac/run_leader_teleop.py \
  --leader_port /dev/ttyACM0 --leader_id my_leader --record_dir /home/teo/so101/runtime/demos/run_001
```

Use `--preview_only` on the IsaacLab command to freeze the recorded reset pose
for camera/layout calibration instead of driving the sim live.

To evaluate the saved real ACT checkpoint in the same scene:

```bash
cd /home/teo/so101
./.venv/bin/python act/policy_server.py
```

```bash
cd /home/teo/IsaacLab
TERM=xterm-256color DISPLAY=:0 CONDA_PREFIX=/home/teo/miniconda3/envs/env_isaaclab \
  ./isaaclab.sh -p /home/teo/so101/isaac/run_zero_shot.py --steps 300 --save_every 30
```

## Current scope

The scene is a measured first approximation of the real frame: a light desk, white
back wall, SO-101 at the bottom of the image, a small wrapped bar, and an open tan
basket at the top.  Camera pose and object dimensions are exposed as constants in
`scene.py` for side-by-side calibration against the real camera frame.

The next phase is deliberate: keep tightening the camera and layout until the real
and simulated frames align, then collect a larger sim demo set, train, and compare
policy rollouts in sim and real.

## Reinforcement-learning experiment

`rl_env.py` adds a separate, vectorized PPO teacher task for the same scanned bar
and basket.  It gives dense, stage-gated reward for reaching, closing near the bar,
lifting, carrying toward the basket, and finally releasing inside the calibrated
basket interior.  It is deliberately **state-based** for the first experiment:
the policy sees robot/object state, not RGB.  This is the fastest way to validate
that the grasp collision, reset, and reward specification can learn the task.
Only the bar is randomized at reset (currently ±5 cm in X and Y); the robot,
basket, camera, desk, and enclosure remain fixed to preserve the real-world match.

First open one visible environment and check that it resets cleanly:

```bash
TERM=xterm-256color DISPLAY=:0 CONDA_PREFIX=/home/teo/miniconda3/envs/env_isaaclab \
  /home/teo/IsaacLab/isaaclab.sh -p /home/teo/so101/rl/check_rl_env.py --steps 180 --device cuda:0
```

Then start conservatively with 16 parallel environments (not the 64-env default)
and watch VRAM/RAM before increasing it:

```bash
CONDA_PREFIX=/home/teo/miniconda3/envs/env_isaaclab \
  /home/teo/IsaacLab/isaaclab.sh -p /home/teo/so101/rl/train_rl.py \
  --headless --num_envs 16 --max_iterations 1500 --device cuda:0
```

Do not deploy this PPO checkpoint directly on the physical arm.  The next transfer
step is to render successful PPO rollouts through the calibrated 640x480 policy
camera, turn them into synthetic ACT demonstrations, apply camera/physics/domain
randomization, and fine-tune alongside the real bar demonstrations.  This keeps
the real robot interface and camera observation format aligned with the model that
is actually evaluated on hardware.

To watch the newest saved PPO checkpoint in a visible Isaac Sim window after training:

```bash
TERM=xterm-256color DISPLAY=:0 CONDA_PREFIX=/home/teo/miniconda3/envs/env_isaaclab \
  /home/teo/IsaacLab/isaaclab.sh -p /home/teo/so101/rl/play_rl.py --steps 10000 --device cuda:0
```

## Prompted vision to Cartesian skills

The non-ACT proof separates scene understanding from the reusable robot skills:

```text
C270 image -> YOLOE masks -> planar camera geometry -> Cartesian goals
           -> standalone SO-101 IK -> guarded real-arm executor
```

`perceive_scene.py` uses text-prompted YOLOE segmentation. It selected the real
wrapped chocolate bar and wooden basket without task-specific training, while
the geometric filter rejected a low-confidence false bar detection on the robot.
The small model and text encoder are intentionally ignored by Git:

```bash
./.venv/bin/pip install --target .vision_packages --no-deps \
  ultralytics==8.4.112 ultralytics-thop==2.0.18
./.venv/bin/pip install --target .vision_packages --no-deps \
  ftfy==6.3.1 git+https://github.com/ultralytics/CLIP.git
```

Run perception, create the current matched-workspace calibration, and write a
dry-run skill program:

```bash
PYTHONPATH=/home/teo/so101/.vision_packages \
  ./.venv/bin/python perception/perceive_scene.py --device 0
./.venv/bin/python perception/calibrate_current_workspace.py
./.venv/bin/python code_as_policy/plan_from_scene.py
```

The calibration should be created once from the untouched reference scene, not
recreated after every object movement. `plan_from_scene.py` then applies that
fixed mapping to each new frame. Its safe default uses detected X/Y with the
physically proven wrist orientation. `--orientation-mode vision` is experimental
and must be staged above the object and checked through the wrist camera before
closing.

For a genuinely new fixed camera/table setup, measure at least four
non-collinear points spread around the reachable workspace and fit a new
homography:

```bash
./.venv/bin/python perception/calibrate_workspace.py points.json \
  --output runtime/outputs/new_workspace_calibration.json
```

On 2026-07-30 this pipeline produced a successful physical proof:

- YOLOE localized the live bar at approximately `(-0.187, -0.056) m`.
- Isaac Lab generated a 37-second joint-limited Cartesian plan.
- The executor staged at the last open-gripper waypoint and captured both cameras.
- Detected X/Y plus the proven wrist orientation grasped and lifted the real bar.
- The wrist image confirmed the bar held between both fingers.
- The path was retraced, the bar released, the arm returned to reset, and all
  motor torque was disabled.

The subsequent complete run also succeeded: the robot re-observed the rotated
bar, grasped it, transported it through the Cartesian corridor, released it
inside the detected basket, and returned to reset. The first empty post-drop
return exceeded the tighter forward shoulder-lag threshold and therefore used
the automatic safety retrace; `postdrop_*` phases now use the already validated
return-load envelope. The verified final frame is
`~/Desktop/so101_success_bar_in_basket.png`.

Evidence is saved under `~/Desktop/so101_vision_pick_lift/`. Full automatic
vision-derived yaw and bar-to-basket transport remain separate validation steps;
do not treat the current successful pickup as arbitrary 6-DoF generalization.

### End-to-end vision pick-and-place pipeline

`run_vision_pick_place.py` connects the validated components without weakening
their safety boundaries. By default it captures/detects the scene, writes and
checks the task program, and generates a return-home Cartesian plan with the
standalone SO-101 kinematics layer. It does not open the follower serial port:

```bash
PYTHONPATH=/home/teo/so101/.vision_packages \
  ./.venv/bin/python code_as_policy/run_vision_pick_place.py
```

An existing image can be used for a completely camera-independent dry run with
`--image PATH`. Physical execution is deliberately separate and retains the
real executor's literal confirmation gate, telemetry limits, torque watchdog,
and teardown behavior:

```bash
PYTHONPATH=/home/teo/so101/.vision_packages \
  ./.venv/bin/python code_as_policy/run_vision_pick_place.py \
  --execute --confirm MOVE
```

Real execution is allowed only with the physically proven wrist orientation.
The complete plan releases in the basket
and follows the explicit open-gripper `postdrop_*` return path. After torque is
disabled, the pipeline captures a new front image and verifies that the detected
bar center lies inside the detected basket mask. Annotated before/after evidence
is written to `~/Desktop/so101_vision_pick_place/`.

The default execution rates are `0.50x` in free space, `0.25x` at contact, and
`0.40x` while carrying the object. Override them up to the plan's full bounded
speed with `--speed-scale`, `--contact-speed-scale`, and
`--loaded-speed-scale`. The first Ctrl-C performs a monitored `0.30x` retrace
to the starting pose; a second Ctrl-C immediately freezes the goal and releases
torque. Set the return rate with `--interrupt-return-speed-scale` (maximum
`0.50x`).

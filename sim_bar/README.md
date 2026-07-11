# SO-101 bar pick-and-place: zero-shot simulation evaluation

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
./.venv/bin/python sim_bar/leader_server.py --leader_port /dev/ttyACM0 --leader_id my_leader
```

```bash
# Terminal 2: launch Isaac Sim/Isaac Lab with the nominal bar scene.
cd /home/teo/IsaacLab
TERM=xterm-256color DISPLAY=:0 CONDA_PREFIX=/home/teo/miniconda3/envs/env_isaaclab \
  ./isaaclab.sh -p /home/teo/so101/sim_bar/run_leader_teleop.py \
  --leader_port /dev/ttyACM0 --leader_id my_leader --record_dir /home/teo/so101/sim_bar/demos/run_001
```

Use `--preview_only` on the IsaacLab command to freeze the recorded reset pose
for camera/layout calibration instead of driving the sim live.

To evaluate the saved real ACT checkpoint in the same scene:

```bash
cd /home/teo/so101
./.venv/bin/python sim_bar/policy_server.py
```

```bash
cd /home/teo/IsaacLab
TERM=xterm-256color DISPLAY=:0 CONDA_PREFIX=/home/teo/miniconda3/envs/env_isaaclab \
  ./isaaclab.sh -p /home/teo/so101/sim_bar/run_zero_shot.py --steps 300 --save_every 30
```

## Current scope

The scene is a measured first approximation of the real frame: a light desk, white
back wall, SO-101 at the bottom of the image, a small wrapped bar, and an open tan
basket at the top.  Camera pose and object dimensions are exposed as constants in
`scene.py` for side-by-side calibration against the real camera frame.

The next phase is deliberate: keep tightening the camera and layout until the real
and simulated frames align, then collect a larger sim demo set, train, and compare
policy rollouts in sim and real.

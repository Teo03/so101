# Standalone SO-101 kinematics

This package converts Cartesian targets into the calibrated joint degrees used
by LeRobot. It does not import Isaac, open a serial port, or enable torque.

The three calibrations have separate responsibilities:

- LeRobot's `my_follower.json`: raw servo ticks to calibrated joint degrees.
- `runtime/outputs/current_workspace_calibration.json`: camera pixels to desk XY.
- The real-task anchors in `so101.py`: desk XYZ to the official SO-101 URDF frame.

The included anchors cover the physically proven bar-to-basket corridor. Add
measured real anchor poses before expecting accurate IK in a substantially
different part of the workspace.

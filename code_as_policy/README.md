# Code as Policy

High-level task programs and orchestration only.

- `plan_from_scene.py` converts grounded detections into an inspectable skill program.
- `run_vision_pick_place.py` coordinates perception, Isaac IK compilation, guarded
  hardware execution, and result verification.

This layer must call bounded APIs; it does not own serial communication or raw
30 Hz motor control.

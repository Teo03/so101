# Robot skills

Task scripts compose these bounded skills instead of implementing motor control.
`PickPlaceTask` currently compiles pick, vertical lift, high transport, place,
retreat, and return-home into the plan format consumed by
`hardware/run_real_cartesian.py`.

Example task script:

```python
from pathlib import Path

from skills import PickPlaceTask, build_pick_place_plan

task = PickPlaceTask(
    pick_xyz_m=(-0.19, -0.05, 0.0285),
    drop_xyz_m=(0.04, 0.14, 0.105),
)
build_pick_place_plan(task, Path("runtime/outputs/my_task.npz"))
```

Writing the plan never moves the robot. Execution remains a separate guarded
command requiring `--confirm MOVE`.

# SO-101 sim-to-real workspace

This repository combines several complementary ways to control the same SO-101
robot. They are intentionally separate layers rather than competing projects:

```text
language/task program (Code as Policy)
              |
     perception + geometry
              |
   +----------+-----------+
   |                      |
learned ACT skill   Cartesian skill
   |                      |
   +----------+-----------+
              |
       guarded real executor
```

Use the unified command index instead of remembering individual script names:

```bash
./.venv/bin/python -m tools.cli list
```

The source map and subsystem boundaries are documented in
[`CODEMAP.md`](CODEMAP.md). Operational commands and validated hardware notes
remain in [`OPERATIONS.md`](OPERATIONS.md). The private
`lerobot/` submodule contains the independently maintained ACT and DAgger
implementation.

## Repository boundaries

- `code_as_policy/`: task programs and orchestration.
- `perception/`: scene detection and workspace calibration.
- `hardware/`: guarded real-arm and leader interfaces.
- `kinematics/`: standalone SO-101 FK/IK and workspace-frame registration.
- `skills/`: reusable task-to-trajectory building blocks.
- `act/`: learned ACT policy service.
- `isaac/`: simulation scene, teleoperation, IK, and validation.
- `rl/`: isolated PPO teacher experiment.
- `assets/scans/`: scanned GLB/USD objects and their textures.
- `runtime/`: ignored outputs and demonstrations.
- `lerobot/`: private pinned LeRobot submodule; do not mix project scripts into it.
- `runtime/outputs/`, `runtime/demos/`, `outputs/`, logs, caches, and model weights:
  runtime data ignored by Git.

## Safe default

Planning is offline and does not open the serial port. Real motion still requires the
underlying executor's explicit `--confirm MOVE` gate; the unified CLI does not
bypass it.

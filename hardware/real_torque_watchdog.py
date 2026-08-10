#!/usr/bin/env python3
"""Disable SO-101 torque if a physical-motion controller disappears."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from run_real_cartesian import (
    DEFAULT_CALIBRATION,
    DEFAULT_PORT,
    load_calibration,
    make_bus,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller-pid", type=int, required=True)
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    args = parser.parse_args()

    while Path(f"/proc/{args.controller_pid}").exists():
        time.sleep(0.1)

    # The dead controller may have left the serial device in teardown briefly.
    # Retry independently; never send a position command from this process.
    last_error: Exception | None = None
    for _ in range(30):
        bus = make_bus(args.port, load_calibration(args.calibration))
        try:
            bus.connect(handshake=True)
            bus.disable_torque(num_retry=3)
            print(
                f"[watchdog] controller {args.controller_pid} exited; torque disabled",
                flush=True,
            )
            return
        except Exception as error:
            last_error = error
            time.sleep(0.2)
        finally:
            if bus.is_connected:
                bus.disconnect(disable_torque=False)
    raise RuntimeError(f"Could not disable torque after controller exit: {last_error}")


if __name__ == "__main__":
    main()

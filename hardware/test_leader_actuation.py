#!/usr/bin/env python3
"""Perform a tiny, reversible actuation test on one SO-101 leader joint."""

from __future__ import annotations

import argparse
import time

from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--leader-id", default="my_leader")
    parser.add_argument("--joint", default="shoulder_pan")
    parser.add_argument("--delta-deg", type=float, default=2.0)
    parser.add_argument("--step-deg", type=float, default=0.25)
    parser.add_argument("--period-s", type=float, default=0.10)
    args = parser.parse_args()

    if not 0 < abs(args.delta_deg) <= 3.0:
        raise ValueError("delta-deg must be non-zero and at most 3 degrees")
    if not 0 < args.step_deg <= 0.5:
        raise ValueError("step-deg must be in (0, 0.5]")

    leader = SOLeader(SOLeaderTeleopConfig(port=args.port, id=args.leader_id, use_degrees=True))
    torque_enabled = False
    try:
        leader.connect()
        start = {k.removesuffix(".pos"): v for k, v in leader.get_action().items()}
        if args.joint not in start:
            raise KeyError(f"Unknown joint {args.joint!r}")

        print("Current calibrated leader pose:")
        for name, value in start.items():
            print(f"  {name:14s} {value:8.2f}")

        # Seed every goal with the measured pose before torque is enabled. This
        # prevents stale servo goals from pulling any joint unexpectedly.
        leader.bus.sync_write("Goal_Position", start)
        leader.bus.enable_torque(args.joint, num_retry=3)
        torque_enabled = True
        time.sleep(0.5)

        target = start[args.joint] + args.delta_deg
        count = max(1, int(abs(args.delta_deg) / args.step_deg))
        print(f"Moving only {args.joint} by {args.delta_deg:+.2f} degrees...")
        for index in range(1, count + 1):
            goal = start[args.joint] + (target - start[args.joint]) * index / count
            leader.bus.write("Goal_Position", args.joint, goal)
            time.sleep(args.period_s)

        time.sleep(1.0)
        measured = leader.get_action()[f"{args.joint}.pos"]
        if abs(measured - target) > 1.0:
            raise RuntimeError(
                f"Tracking error too large: target={target:.2f}, measured={measured:.2f}"
            )
        print(f"Reached {measured:.2f} degrees; returning to start...")

        for index in range(1, count + 1):
            goal = target + (start[args.joint] - target) * index / count
            leader.bus.write("Goal_Position", args.joint, goal)
            time.sleep(args.period_s)

        final = leader.get_action()[f"{args.joint}.pos"]
        print(f"Final {args.joint}: {final:.2f} degrees (start {start[args.joint]:.2f})")
    finally:
        if leader.is_connected:
            if torque_enabled:
                leader.bus.disable_torque(args.joint, num_retry=3)
            leader.disconnect()
        print("Leader torque disabled.")


if __name__ == "__main__":
    main()

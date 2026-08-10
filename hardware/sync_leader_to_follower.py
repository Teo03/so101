#!/usr/bin/env python3
"""Safely synchronize an SO-101 leader pose to an SO-101 follower pose."""

from __future__ import annotations

import argparse
import math
import time

from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig


JOINT_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

# Calibrated-space limits on how far an unattended synchronization may travel.
MAX_INITIAL_ERROR = {
    "shoulder_pan": 120.0,
    "shoulder_lift": 120.0,
    "elbow_flex": 120.0,
    "wrist_flex": 120.0,
    "wrist_roll": 180.0,
    "gripper": 100.0,
}


def read_pose(bus) -> dict[str, float]:
    pose = bus.sync_read("Present_Position", num_retry=3)
    if set(pose) != set(JOINT_ORDER):
        raise RuntimeError(f"Unexpected motor keys: {sorted(pose)}")
    if not all(math.isfinite(float(value)) for value in pose.values()):
        raise RuntimeError(f"Non-finite position in pose: {pose}")
    return {name: float(pose[name]) for name in JOINT_ORDER}


def print_comparison(leader_pose: dict[str, float], follower_pose: dict[str, float]) -> None:
    print("Joint             Leader    Follower       Delta")
    for joint in JOINT_ORDER:
        delta = follower_pose[joint] - leader_pose[joint]
        print(f"{joint:14s} {leader_pose[joint]:9.2f} {follower_pose[joint]:11.2f} {delta:+11.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leader-port", required=True)
    parser.add_argument("--follower-port", required=True)
    parser.add_argument("--leader-id", default="my_leader")
    parser.add_argument("--follower-id", default="my_follower")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--step", type=float, default=0.5, help="Maximum calibrated units per command")
    parser.add_argument("--period-s", type=float, default=0.10)
    parser.add_argument("--settle-s", type=float, default=2.0)
    parser.add_argument("--tolerance", type=float, default=3.0)
    args = parser.parse_args()

    if not 0 < args.step <= 1.0:
        raise ValueError("--step must be in (0, 1]")
    if not 0.05 <= args.period_s <= 0.5:
        raise ValueError("--period-s must be in [0.05, 0.5]")

    leader = SOLeader(SOLeaderTeleopConfig(port=args.leader_port, id=args.leader_id, use_degrees=True))
    follower = SOFollower(
        SOFollowerRobotConfig(port=args.follower_port, id=args.follower_id, use_degrees=True, cameras={})
    )
    enabled_joints: list[str] = []
    try:
        # Direct bus connections intentionally avoid configure(), torque changes,
        # camera startup, or any Goal_Position write on the follower.
        follower.bus.connect()
        leader.bus.connect()
        follower_pose = read_pose(follower.bus)
        leader_pose = read_pose(leader.bus)
        print_comparison(leader_pose, follower_pose)

        excessive = {
            joint: follower_pose[joint] - leader_pose[joint]
            for joint in JOINT_ORDER
            if abs(follower_pose[joint] - leader_pose[joint]) > MAX_INITIAL_ERROR[joint]
        }
        if excessive:
            raise RuntimeError(f"Refusing excessive synchronization travel: {excessive}")

        # Exercise calibration conversion before any torque or motion.
        leader.bus._unnormalize(
            {leader.bus.motors[joint].id: follower_pose[joint] for joint in JOINT_ORDER}
        )
        if not args.execute:
            print("DRY RUN PASSED: targets are finite, mapped, and inside the travel guard.")
            return

        print("Executing one joint at a time. The follower remains read-only.")
        for joint in JOINT_ORDER:
            target = follower_pose[joint]
            current = read_pose(leader.bus)[joint]
            distance = target - current
            if abs(distance) <= args.tolerance:
                print(f"{joint}: already aligned ({distance:+.2f})")
                continue

            # Seed the exact measured position before enabling this joint.
            leader.bus.write("Goal_Position", joint, current, num_retry=3)
            leader.bus.enable_torque(joint, num_retry=3)
            enabled_joints.append(joint)
            for correction_pass in range(1, 4):
                current = read_pose(leader.bus)[joint]
                distance = target - current
                count = max(1, math.ceil(abs(distance) / args.step))
                print(
                    f"{joint} pass {correction_pass}: {current:.2f} -> {target:.2f} "
                    f"in {count} increments"
                )
                for index in range(1, count + 1):
                    goal = current + distance * index / count
                    leader.bus.write("Goal_Position", joint, goal, num_retry=3)
                    time.sleep(args.period_s)

                time.sleep(args.settle_s)
                measured = read_pose(leader.bus)[joint]
                error = target - measured
                print(f"{joint}: measured {measured:.2f}, error {error:+.2f}")
                if abs(error) <= args.tolerance:
                    break
            else:
                raise RuntimeError(f"{joint} failed to track target after 3 correction passes")

        final_pose = read_pose(leader.bus)
        print_comparison(final_pose, follower_pose)
        print("SYNCHRONIZATION PASSED.")
    finally:
        if leader.bus.is_connected:
            for joint in reversed(enabled_joints):
                try:
                    leader.bus.disable_torque(joint, num_retry=3)
                except Exception as exc:
                    print(f"WARNING: failed to disable {joint}: {exc}")
            leader.bus.disconnect(disable_torque=False)
        if follower.bus.is_connected:
            # The follower was never enabled or commanded by this script.
            follower.bus.disconnect(disable_torque=False)
        print("Leader synchronization torque disabled; follower was read-only.")


if __name__ == "__main__":
    main()

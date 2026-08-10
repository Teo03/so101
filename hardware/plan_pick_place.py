#!/usr/bin/env python3
"""Compile a Cartesian pick/place task with standalone SO-101 IK."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skills import PickPlaceTask, build_pick_place_plan  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pick-x", type=float, required=True)
    parser.add_argument("--pick-y", type=float, required=True)
    parser.add_argument("--pick-z", type=float, required=True)
    parser.add_argument("--drop-x", type=float, required=True)
    parser.add_argument("--drop-y", type=float, required=True)
    parser.add_argument("--drop-z", type=float, required=True)
    parser.add_argument("--transport-z", type=float, default=0.175)
    parser.add_argument("--output", type=Path, default=ROOT / "runtime/outputs/real_cartesian_plan.npz")
    parser.add_argument("--no-return-home", action="store_true")
    args = parser.parse_args()
    task = PickPlaceTask(
        pick_xyz_m=(args.pick_x, args.pick_y, args.pick_z),
        drop_xyz_m=(args.drop_x, args.drop_y, args.drop_z),
        transport_z_m=args.transport_z,
        return_home=not args.no_return_home,
    )
    path = build_pick_place_plan(task, args.output)
    print(f"[plan] wrote standalone real-robot plan to {path}")


if __name__ == "__main__":
    main()

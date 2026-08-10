#!/usr/bin/env python3
"""Fit a planar pixel-to-robot homography for a new fixed tabletop setup.

Input JSON example:
{
  "points": [
    {"pixel": [102, 315], "robot_xy_m": [-0.20, -0.10]},
    {"pixel": [500, 300], "robot_xy_m": [ 0.20, -0.10]},
    {"pixel": [460, 120], "robot_xy_m": [ 0.20,  0.25]},
    {"pixel": [150, 130], "robot_xy_m": [-0.20,  0.25]}
  ]
}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perception.workspace_geometry import fit_homography  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("correspondences", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plane-z-m", type=float, default=0.0)
    parser.add_argument("--max-error-mm", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = json.loads(args.correspondences.read_text())
    points = source["points"]
    pixels = np.asarray([point["pixel"] for point in points], dtype=np.float64)
    robot_xy = np.asarray([point["robot_xy_m"] for point in points], dtype=np.float64)
    homography, inliers = fit_homography(pixels, robot_xy)
    predicted = cv2.perspectiveTransform(pixels.reshape(-1, 1, 2), homography).reshape(-1, 2)
    error_mm = np.linalg.norm(predicted - robot_xy, axis=1) * 1000.0
    if not np.all(inliers):
        raise RuntimeError(f"Calibration rejected point indices: {np.flatnonzero(~inliers).tolist()}")
    if float(error_mm.max()) > args.max_error_mm:
        raise RuntimeError(
            f"Calibration error {error_mm.max():.2f} mm exceeds {args.max_error_mm:.2f} mm"
        )
    payload = {
        "format": "so101-planar-workspace-v1",
        "method": "measured_table_correspondences",
        "plane_z_m": args.plane_z_m,
        "pixel_to_robot_xy_homography": homography.tolist(),
        "calibration_points": points,
        "point_error_mm": error_mm.tolist(),
        "max_error_mm": float(error_mm.max()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "error_mm": error_mm.tolist()}, indent=2))


if __name__ == "__main__":
    main()

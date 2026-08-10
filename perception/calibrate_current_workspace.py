#!/usr/bin/env python3
"""Create the current workspace's approximate planar calibration.

The calibrated Isaac camera supplies the projective shape.  Two real-image
landmarks (bar and basket) align that model to the physical C270 frame.  This
is sufficient for a no-motion proof in the current matched workspace.  A new
camera mount should use four or more measured table points instead.
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

from perception.detect_real_bar import BAR_WORLD_Z_M, project_xy  # noqa: E402


DEFAULT_SCENE = ROOT / "runtime/outputs/scene_observation.json"
DEFAULT_OUTPUT = ROOT / "runtime/outputs/current_workspace_calibration.json"
BAR_XY = np.asarray((-0.190, -0.050), dtype=np.float64)
BASKET_XY = np.asarray((0.120, 0.200), dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def similarity_from_two_pairs(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_delta = source[1] - source[0]
    target_delta = target[1] - target[0]
    scale = np.linalg.norm(target_delta) / np.linalg.norm(source_delta)
    source_angle = np.arctan2(source_delta[1], source_delta[0])
    target_angle = np.arctan2(target_delta[1], target_delta[0])
    angle = target_angle - source_angle
    rotation = np.asarray(
        ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))),
        dtype=np.float64,
    )
    translation = target[0] - scale * rotation @ source[0]
    matrix = np.eye(3, dtype=np.float64)
    matrix[:2, :2] = scale * rotation
    matrix[:2, 2] = translation
    return matrix


def main() -> None:
    args = parse_args()
    scene = json.loads(args.scene.read_text())
    observed = np.asarray(
        (
            scene["selected"]["bar"]["center_px"],
            scene["selected"]["basket"]["center_px"],
        ),
        dtype=np.float64,
    )
    world = np.asarray((BAR_XY, BASKET_XY), dtype=np.float64)
    projected = np.asarray([project_xy(point) for point in world], dtype=np.float64)
    alignment = similarity_from_two_pairs(projected, observed)

    # Fit the exact projective mapping of the matched pinhole model at the bar
    # plane, then align it to the real image with the two physical landmarks.
    grid_world = np.asarray(
        [(x, y) for x in np.linspace(-0.26, 0.40, 5) for y in np.linspace(-0.25, 0.42, 5)],
        dtype=np.float64,
    )
    grid_projected = np.asarray([project_xy(point) for point in grid_world], dtype=np.float64)
    homogeneous = np.column_stack((grid_projected, np.ones(len(grid_projected))))
    aligned_pixels = (alignment @ homogeneous.T).T[:, :2]
    robot_to_pixel, _ = cv2.findHomography(grid_world, aligned_pixels, method=0)
    pixel_to_robot = np.linalg.inv(robot_to_pixel)
    pixel_to_robot /= pixel_to_robot[2, 2]

    check = cv2.perspectiveTransform(observed.reshape(-1, 1, 2), pixel_to_robot).reshape(-1, 2)
    residual_mm = np.linalg.norm(check - world, axis=1) * 1000.0
    bar = scene["selected"]["bar"]
    angle = np.deg2rad(bar["long_axis_angle_deg"])
    direction = np.asarray((np.cos(angle), np.sin(angle)), dtype=np.float64)
    half_length = max(10.0, float(bar["size_px"][0]) * 0.35)
    endpoints = np.stack((observed[0] - half_length * direction, observed[0] + half_length * direction))
    endpoints_world = cv2.perspectiveTransform(
        endpoints.reshape(-1, 1, 2), pixel_to_robot
    ).reshape(-1, 2)
    bar_axis = endpoints_world[1] - endpoints_world[0]
    reference_yaw_deg = float(np.degrees(np.arctan2(bar_axis[1], bar_axis[0])))
    payload = {
        "format": "so101-planar-workspace-v1",
        "method": "matched_sim_camera_plus_two_real_landmarks",
        "warning": (
            "Approximate current-workspace calibration. Before moving hardware in a new "
            "camera setup, replace it with >=4 measured table correspondences."
        ),
        "plane_z_m": BAR_WORLD_Z_M,
        "pixel_to_robot_xy_homography": pixel_to_robot.tolist(),
        "landmarks": [
            {"name": "bar", "pixel": observed[0].tolist(), "robot_xy_m": BAR_XY.tolist()},
            {"name": "basket", "pixel": observed[1].tolist(), "robot_xy_m": BASKET_XY.tolist()},
        ],
        "landmark_residual_mm": residual_mm.tolist(),
        "bar_reference_yaw_deg": reference_yaw_deg,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "residual_mm": residual_mm.tolist()}, indent=2))


if __name__ == "__main__":
    main()

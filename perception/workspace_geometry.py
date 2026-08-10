"""Planar camera-to-robot geometry shared by perception and task planning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class PlanarWorkspace:
    """Map image pixels to XY metres in the robot base frame."""

    pixel_to_robot_xy: np.ndarray
    metadata: dict[str, object]

    @classmethod
    def load(cls, path: Path) -> "PlanarWorkspace":
        payload = json.loads(path.read_text())
        matrix = np.asarray(payload["pixel_to_robot_xy_homography"], dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError(f"Invalid homography in {path}")
        return cls(matrix, payload)

    def pixel_to_xy(self, pixel: list[float] | np.ndarray) -> np.ndarray:
        points = np.asarray(pixel, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(points, self.pixel_to_robot_xy).reshape(-1, 2)

    def pixel_axis_to_world_yaw(self, center: list[float], angle_deg: float, length_px: float) -> float:
        angle = np.deg2rad(angle_deg)
        half = max(10.0, length_px * 0.35)
        direction = np.asarray((np.cos(angle), np.sin(angle)), dtype=np.float64)
        endpoints = np.stack((np.asarray(center) - half * direction, np.asarray(center) + half * direction))
        world = self.pixel_to_xy(endpoints)
        delta = world[1] - world[0]
        return float(np.arctan2(delta[1], delta[0]))


def fit_homography(
    image_points_px: np.ndarray,
    robot_points_xy_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a robust pixel-to-robot homography from at least four point pairs."""
    image_points = np.asarray(image_points_px, dtype=np.float64)
    robot_points = np.asarray(robot_points_xy_m, dtype=np.float64)
    if image_points.shape != robot_points.shape or image_points.ndim != 2 or image_points.shape[1] != 2:
        raise ValueError("Image and robot calibration points must both have shape (N, 2)")
    if len(image_points) < 4:
        raise ValueError("A full homography requires at least four non-collinear point pairs")
    matrix, inliers = cv2.findHomography(image_points, robot_points, cv2.RANSAC, 0.004)
    if matrix is None:
        raise RuntimeError("OpenCV could not fit a homography to the supplied calibration points")
    return matrix, inliers.reshape(-1).astype(bool)

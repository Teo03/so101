#!/usr/bin/env python3
"""Detect the real chocolate bar and estimate pose change from a known grasp.

The successful fixed-pose grasp frame is the calibration anchor.  A local
camera-to-table Jacobian from the matched Isaac camera converts pixel motion to
metric XY motion.  Orientation is likewise reported as a delta from the
successful anchor, avoiding an unnecessary absolute hand-eye calibration for
small motions inside the validated pickup region.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANCHOR = Path.home() / "Desktop/so101_before_grasp_front.png"
DEFAULT_CAPTURE = ROOT / "runtime/outputs/real_bar_live.png"
DEFAULT_ANNOTATED = Path.home() / "Desktop/so101_detected_bar.png"

ANCHOR_WORLD_XY_M = np.asarray((-0.190, -0.050), dtype=np.float64)
BAR_WORLD_Z_M = 0.0305
ROI = (10, 190, 210, 380)  # x0, y0, x1, y1
DARK_THRESHOLD = 105

# The local projection is calculated from the calibrated Isaac policy camera
# at the bar plane.  Its absolute pixel origin differs slightly from the real
# camera, which is why the successful real image supplies the anchor point.
CAMERA_EYE_M = np.asarray((-0.02, -0.55, 0.50), dtype=np.float64)
CAMERA_TARGET_M = np.asarray((0.03, 0.08, 0.04), dtype=np.float64)
FOCAL_LENGTH_MM = 26.0
HORIZONTAL_APERTURE_MM = 20.955
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default="/dev/video0")
    parser.add_argument("--image", type=Path, help="Analyze this image instead of capturing.")
    parser.add_argument("--anchor", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--annotated", type=Path, default=DEFAULT_ANNOTATED)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=ROOT / "runtime/outputs/real_bar_pose.json",
    )
    return parser.parse_args()


def capture(camera: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "v4l2",
            "-video_size",
            f"{IMAGE_WIDTH}x{IMAGE_HEIGHT}",
            "-framerate",
            "30",
            "-i",
            camera,
            "-frames:v",
            "1",
            "-y",
            str(output),
        ],
        check=True,
    )


def normalize_axis_angle_deg(angle_deg: float) -> float:
    return float((angle_deg + 90.0) % 180.0 - 90.0)


def detect(image: np.ndarray) -> dict[str, object]:
    if image is None or image.shape[:2] != (IMAGE_HEIGHT, IMAGE_WIDTH):
        raise RuntimeError(f"Expected a {IMAGE_WIDTH}x{IMAGE_HEIGHT} camera image")
    x0, y0, x1, y1 = ROI
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    roi_mask = (gray[y0:y1, x0:x1] < DARK_THRESHOLD).astype(np.uint8) * 255
    roi_mask = cv2.morphologyEx(
        roi_mask,
        cv2.MORPH_CLOSE,
        np.ones((7, 7), dtype=np.uint8),
        iterations=2,
    )
    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, np.ndarray]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if 1800.0 <= area <= 10000.0:
            candidates.append((area, contour))
    if not candidates:
        raise RuntimeError("No bar-sized dark component found inside the safe pickup ROI")
    _, contour = max(candidates, key=lambda item: item[0])
    contour = contour.copy()
    contour[:, 0, 0] += x0
    contour[:, 0, 1] += y0
    center, size, angle = cv2.minAreaRect(contour)
    width, height = size
    if width < height:
        width, height = height, width
        angle += 90.0
    angle = normalize_axis_angle_deg(angle)
    aspect = width / max(height, 1e-6)
    area = float(cv2.contourArea(contour))
    if not 115.0 <= width <= 190.0:
        raise RuntimeError(f"Detected long side is implausible: {width:.1f}px")
    if not 30.0 <= height <= 80.0:
        raise RuntimeError(f"Detected short side is implausible: {height:.1f}px")
    if aspect < 2.3:
        raise RuntimeError(f"Detected component is not bar-shaped: aspect={aspect:.2f}")
    return {
        "center_px": np.asarray(center, dtype=np.float64),
        "long_axis_angle_deg": angle,
        "size_px": np.asarray((width, height), dtype=np.float64),
        "area_px": area,
        "box_px": cv2.boxPoints((center, (width, height), angle)),
    }


def project_xy(point_xy: np.ndarray) -> np.ndarray:
    point = np.asarray((point_xy[0], point_xy[1], BAR_WORLD_Z_M), dtype=np.float64)
    forward = CAMERA_TARGET_M - CAMERA_EYE_M
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray((0.0, 0.0, 1.0)))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    relative = point - CAMERA_EYE_M
    camera = np.asarray(
        (relative @ right, relative @ up, relative @ forward),
        dtype=np.float64,
    )
    focal_px = IMAGE_WIDTH * FOCAL_LENGTH_MM / HORIZONTAL_APERTURE_MM
    return np.asarray(
        (
            IMAGE_WIDTH / 2.0 + focal_px * camera[0] / camera[2],
            IMAGE_HEIGHT / 2.0 - focal_px * camera[1] / camera[2],
        )
    )


def local_pixel_jacobian() -> np.ndarray:
    epsilon = 0.001
    origin = project_xy(ANCHOR_WORLD_XY_M)
    dx = (project_xy(ANCHOR_WORLD_XY_M + (epsilon, 0.0)) - origin) / epsilon
    dy = (project_xy(ANCHOR_WORLD_XY_M + (0.0, epsilon)) - origin) / epsilon
    return np.column_stack((dx, dy))


def world_axis_angle(pixel_angle_deg: float, inverse_jacobian: np.ndarray) -> float:
    angle = np.deg2rad(pixel_angle_deg)
    pixel_axis = np.asarray((np.cos(angle), np.sin(angle)), dtype=np.float64)
    world_axis = inverse_jacobian @ pixel_axis
    return float(np.arctan2(world_axis[1], world_axis[0]))


def annotate(
    image: np.ndarray,
    detection: dict[str, object],
    world_xy: np.ndarray,
    yaw_delta_deg: float,
    output: Path,
) -> None:
    rendered = image.copy()
    box = np.rint(detection["box_px"]).astype(np.int32)
    center = np.rint(detection["center_px"]).astype(np.int32)
    cv2.polylines(rendered, [box], True, (0, 255, 0), 2)
    cv2.drawMarker(rendered, tuple(center), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
    label = (
        f"x={world_xy[0]:+.3f} y={world_xy[1]:+.3f} "
        f"yaw_delta={yaw_delta_deg:+.1f}deg"
    )
    cv2.putText(rendered, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
    cv2.putText(rendered, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), rendered):
        raise RuntimeError(f"Could not write {output}")


def main() -> None:
    args = parse_args()
    if not args.anchor.exists():
        raise FileNotFoundError(f"Successful grasp anchor not found: {args.anchor}")
    image_path = args.image
    if image_path is None:
        capture(args.camera, args.capture)
        image_path = args.capture
    image = cv2.imread(str(image_path))
    anchor_image = cv2.imread(str(args.anchor))
    live = detect(image)
    anchor = detect(anchor_image)

    jacobian = local_pixel_jacobian()
    inverse_jacobian = np.linalg.inv(jacobian)
    pixel_delta = live["center_px"] - anchor["center_px"]
    world_delta = inverse_jacobian @ pixel_delta
    world_xy = ANCHOR_WORLD_XY_M + world_delta
    anchor_yaw = world_axis_angle(anchor["long_axis_angle_deg"], inverse_jacobian)
    live_yaw = world_axis_angle(live["long_axis_angle_deg"], inverse_jacobian)
    yaw_delta = float((live_yaw - anchor_yaw + np.pi / 2.0) % np.pi - np.pi / 2.0)

    if np.any(np.abs(world_delta) > np.asarray((0.055, 0.055))):
        raise RuntimeError(
            f"Detected bar moved outside the validated local region: delta={world_delta}"
        )
    if abs(np.degrees(yaw_delta)) > 40.0:
        raise RuntimeError(
            f"Detected bar rotation exceeds the validated range: {np.degrees(yaw_delta):.1f}deg"
        )

    payload = {
        "image": str(image_path),
        "anchor": str(args.anchor),
        "center_px": live["center_px"].tolist(),
        "anchor_center_px": anchor["center_px"].tolist(),
        "pixel_delta": pixel_delta.tolist(),
        "size_px": live["size_px"].tolist(),
        "long_axis_angle_deg": live["long_axis_angle_deg"],
        "anchor_long_axis_angle_deg": anchor["long_axis_angle_deg"],
        "world_xy_m": world_xy.tolist(),
        "world_delta_m": world_delta.tolist(),
        "yaw_delta_deg": float(np.degrees(yaw_delta)),
        "local_pixel_jacobian_px_per_m": jacobian.tolist(),
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2) + "\n")
    annotate(image, live, world_xy, payload["yaw_delta_deg"], args.annotated)
    print(json.dumps(payload, indent=2), flush=True)
    print(f"[vision] annotated={args.annotated} pose={args.json_output}", flush=True)


if __name__ == "__main__":
    main()

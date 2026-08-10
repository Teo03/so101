"""Standalone SO-101 FK/IK in the calibrated real-workspace frame.

The motor bus supplies calibrated joint angles in degrees.  This module does
not read raw servo ticks and does not open hardware; it only converts between
Cartesian poses and those calibrated angles.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[1]
LEROBOT_SRC = ROOT / "lerobot/src"
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.model import RobotKinematics  # noqa: E402


ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
SAFE_MIN = np.asarray([-66.0, -108.0, -99.0, 38.0, -76.0])
SAFE_MAX = np.asarray([20.0, 73.0, 98.0, 103.0, 49.0])

# Four real-task anchors previously executed successfully.  They register the
# camera/workspace coordinates to the official SO-101 URDF frame.  This is not
# motor calibration; it is the Cartesian frame transform for this fixed desk.
_WORKSPACE_ANCHORS = np.asarray(
    [
        [-0.190, -0.050, 0.028535],
        [-0.190, -0.050, 0.175000],
        [0.040, 0.140, 0.105000],
        [0.040, 0.140, 0.175000],
        [-0.188891, -0.053531, 0.028535],
        [-0.188891, -0.053531, 0.175000],
    ],
    dtype=np.float64,
)
_JOINT_ANCHORS = np.asarray(
    [
        [-48.90, 47.48, -40.85, 87.51, -32.52],
        [-48.88, 16.59, -46.53, 100.00, -32.52],
        [1.49, 56.78, -78.45, 75.79, 17.01],
        [1.49, 48.98, -86.52, 79.88, 17.01],
        [-49.33, 47.43, -40.84, 88.66, -32.78],
        [-49.31, 15.08, -44.37, 100.00, -32.78],
    ],
    dtype=np.float64,
)


class SO101Kinematics:
    """Position IK constrained to the physically demonstrated joint envelope."""

    def __init__(self, urdf_path: Path | None = None) -> None:
        self.urdf_path = urdf_path or ROOT / "assets/robot_models/so101_new_calib.urdf"
        self.model = RobotKinematics(
            str(self.urdf_path),
            target_frame_name="gripper_frame_link",
            joint_names=ARM_JOINT_NAMES,
        )
        anchor_positions = np.stack([self.forward(q) for q in _JOINT_ANCHORS])
        homogeneous = np.column_stack((_WORKSPACE_ANCHORS, np.ones(len(_WORKSPACE_ANCHORS))))
        self._workspace_to_urdf = np.linalg.lstsq(
            homogeneous, anchor_positions, rcond=None
        )[0]

    def forward(self, joints_deg: np.ndarray) -> np.ndarray:
        """Return gripper-frame XYZ in the official URDF base frame."""
        return self.model.forward_kinematics(np.asarray(joints_deg, dtype=float))[:3, 3].copy()

    def workspace_to_urdf(self, xyz_m: np.ndarray) -> np.ndarray:
        point = np.append(np.asarray(xyz_m, dtype=float), 1.0)
        return point @ self._workspace_to_urdf

    def inverse_position(
        self,
        xyz_m: np.ndarray,
        seed_deg: np.ndarray,
        *,
        fixed_wrist_roll_deg: float | None = None,
        tolerance_m: float = 0.012,
    ) -> np.ndarray:
        """Solve a workspace XYZ target near a known safe real configuration."""
        seed = np.asarray(seed_deg, dtype=float)[:5].copy()
        target = self.workspace_to_urdf(xyz_m)
        if fixed_wrist_roll_deg is None:
            active = np.arange(5)
        else:
            seed[4] = fixed_wrist_roll_deg
            active = np.arange(4)

        def residual(active_values: np.ndarray) -> np.ndarray:
            candidate = seed.copy()
            candidate[active] = active_values
            position_error = (self.forward(candidate) - target) / 0.01
            # Position-only IK is redundant.  A light posture cost keeps the
            # answer on the demonstrated elbow/wrist branch supplied by the
            # task skill instead of selecting an equally valid folded pose.
            posture_error = (active_values - seed[active]) / 12.0
            return np.concatenate((position_error, posture_error))

        result = least_squares(
            residual,
            seed[active],
            bounds=(SAFE_MIN[active], SAFE_MAX[active]),
            max_nfev=160,
            xtol=1e-9,
            ftol=1e-9,
            gtol=1e-9,
        )
        solved = seed.copy()
        solved[active] = result.x
        error_m = float(np.linalg.norm(self.forward(solved) - target))
        if error_m > tolerance_m:
            raise RuntimeError(
                f"SO-101 IK could not reach {np.round(xyz_m, 4)}; error={error_m:.4f} m"
            )
        return solved.astype(np.float32)

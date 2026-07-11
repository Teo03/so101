"""Pure SO-101 unit conversions shared by the simulation client and policy bridge."""

from __future__ import annotations

import numpy as np

# This is the calibrated mapping supplied with NVIDIA's SO-101 workshop USD.
# LeRobot records the five arm motors in [-100, 100] and the gripper in [0, 100].
JOINT_MINS_DEG = np.asarray([-110.0, -100.0, -100.0, -95.0, -160.0, -10.0], dtype=np.float32)
JOINT_MAXS_DEG = np.asarray([110.0, 100.0, 90.0, 95.0, 160.0, 100.0], dtype=np.float32)


def lerobot_to_sim_radians(values: np.ndarray) -> np.ndarray:
    """Convert a six-value LeRobot SO-101 action to Isaac USD joint radians."""
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (6,):
        raise ValueError(f"Expected six SO-101 values, got {values.shape}.")
    normalized = np.empty(6, dtype=np.float32)
    normalized[:5] = (values[:5] + 100.0) / 200.0
    normalized[5] = values[5] / 100.0
    degrees = JOINT_MINS_DEG + normalized * (JOINT_MAXS_DEG - JOINT_MINS_DEG)
    return np.deg2rad(degrees).astype(np.float32)


def sim_radians_to_lerobot(values: np.ndarray) -> np.ndarray:
    """Convert six Isaac USD joint radians to the model's LeRobot motor-space values."""
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (6,):
        raise ValueError(f"Expected six SO-101 values, got {values.shape}.")
    degrees = np.rad2deg(values)
    normalized = (degrees - JOINT_MINS_DEG) / (JOINT_MAXS_DEG - JOINT_MINS_DEG)
    raw = np.empty(6, dtype=np.float32)
    raw[:5] = normalized[:5] * 200.0 - 100.0
    raw[5] = normalized[5] * 100.0
    return raw

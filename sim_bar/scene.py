"""Nominal Isaac Lab scene for the real SO-101 bar-to-basket task."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils import configclass


WORKSHOP_SOURCE = Path("/home/teo/sim-to-real-so101/Sim-to-Real-SO-101-Workshop/source")
if str(WORKSHOP_SOURCE) not in sys.path:
    sys.path.insert(0, str(WORKSHOP_SOURCE))

from sim_to_real_so101.assets.so101 import S0101_CONTACT_GRASP_CFG  # noqa: E402


# These values are the first metric approximation of the captured real frame.
# Adjust them only after comparing sim_bar/outputs/initial_frame.png with the real frame.
TABLE_SIZE_M = (0.72, 0.82, 0.04)
TABLE_TOP_Z_M = 0.0
# Real layout: the bar starts to the left of the arm and the basket sits back
# and to its right (as seen in outputs/captured_images/opencv__dev_video0.png).
BAR_START_POS_M = (-0.24, -0.05, 0.038)
BASKET_CENTER_M = (0.08, 0.30, 0.08)
# More centered, slightly overhead C270 view to match the real shot.
FRONT_CAMERA_EYE_M = (-0.02, -0.55, 0.50)
FRONT_CAMERA_TARGET_M = (0.03, 0.08, 0.04)

# Median-like reset state from the first frames of the 40 real demonstrations,
# converted with bridge.lerobot_to_sim_radians().  Using NVIDIA's generic pose
# puts the arm far outside the visual/state distribution seen by this ACT model.
REAL_RESET_JOINTS_RAD = {
    "Rotation": -0.0959931,
    # The real servo reaches roughly -106 motor units; the workshop USD hard
    # limit is -100, so this one value is clamped to the simulated limit.
    "Pitch": -1.7440000,
    "Elbow": 1.4547317,
    "Wrist_Pitch": 1.1440634,
    "Wrist_Roll": 0.1117010,
    "Jaw": -0.1380555,
}


def material(color: tuple[float, float, float], roughness: float = 0.65) -> sim_utils.PreviewSurfaceCfg:
    return sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=roughness)


@configclass
class BarPickPlaceSceneCfg(InteractiveSceneCfg):
    """Robot, movable bar, and one RGB observation camera."""

    num_envs = 1
    env_spacing = 2.0

    robot = S0101_CONTACT_GRASP_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos = (0.05, -0.25, 0.0)
    # Human-right-arm view: the shoulder/base is in the foreground and the
    # gripper points forward into the work area toward the basket.
    # 180° yaw: this is the camera pose where the physical gripper points
    # forward into the work area toward the basket.
    robot.init_state.rot = (0.0, 0.0, 0.0, 1.0)
    robot.init_state.joint_pos = REAL_RESET_JOINTS_RAD

    bar = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Bar",
        spawn=sim_utils.CuboidCfg(
            size=(0.120, 0.028, 0.014),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.018),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=material((0.32, 0.16, 0.05), roughness=0.45),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=BAR_START_POS_M),
    )

    front_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/FrontCamera",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=26.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 3.0),
        ),
    )


def _spawn_static_cuboid(
    path: str,
    size: tuple[float, float, float],
    translation: tuple[float, float, float],
    color: tuple[float, float, float],
    *,
    collision: bool = True,
) -> None:
    cfg = sim_utils.CuboidCfg(
        size=size,
        visual_material=material(color),
        collision_props=sim_utils.CollisionPropertiesCfg() if collision else None,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True) if collision else None,
    )
    cfg.func(path, cfg, translation=translation)


def spawn_static_workspace() -> None:
    """Build the desk, three-sided white enclosure, and open-stick basket."""
    _spawn_static_cuboid(
        "/World/Table",
        TABLE_SIZE_M,
        (0.08, 0.08, TABLE_TOP_Z_M - TABLE_SIZE_M[2] / 2),
        (0.66, 0.64, 0.56),
    )
    _spawn_static_cuboid(
        "/World/BackWall",
        (1.40, 0.03, 0.75),
        (0.08, 0.49, 0.365),
        (0.90, 0.91, 0.90),
        collision=False,
    )
    _spawn_static_cuboid(
        "/World/LeftWall",
        (0.03, 1.40, 0.75),
        (-0.30, 0.08, 0.365),
        (0.90, 0.91, 0.90),
        collision=False,
    )
    _spawn_static_cuboid(
        "/World/RightWall",
        (0.03, 1.40, 0.75),
        (0.46, 0.08, 0.365),
        (0.90, 0.91, 0.90),
        collision=False,
    )

    # A visually open basket: thin upright wooden rods on a circular rim.  It is
    # kinematic so the bar can enter without a fragile contact model in this first pass.
    radius = 0.14
    for index in range(28):
        theta = 2.0 * math.pi * index / 28
        x = BASKET_CENTER_M[0] + radius * math.cos(theta)
        y = BASKET_CENTER_M[1] + radius * math.sin(theta)
        _spawn_static_cuboid(
            f"/World/Basket/Rod_{index:02d}",
            (0.007, 0.007, 0.15),
            (x, y, 0.075),
            (0.53, 0.34, 0.17),
            collision=False,
        )
    _spawn_static_cuboid(
        "/World/Basket/Base",
        (0.17, 0.17, 0.012),
        (BASKET_CENTER_M[0], BASKET_CENTER_M[1], 0.006),
        (0.53, 0.34, 0.17),
        collision=False,
    )


class BarPickPlaceScene:
    """Thin wrapper around Isaac Lab scene objects used by the evaluator."""

    def __init__(self, sim: sim_utils.SimulationContext) -> None:
        spawn_static_workspace()
        self.scene = InteractiveScene(BarPickPlaceSceneCfg())
        self.sim = sim

    @property
    def robot(self) -> Articulation:
        return self.scene["robot"]

    @property
    def bar(self) -> RigidObject:
        return self.scene["bar"]

    @property
    def camera(self) -> Camera:
        return self.scene["front_camera"]

    def reset(self) -> None:
        root_state = self.robot.data.default_root_state.clone()
        self.robot.write_root_pose_to_sim(root_state[:, :7])
        self.robot.write_root_velocity_to_sim(root_state[:, 7:])
        self.robot.write_joint_state_to_sim(
            self.robot.data.default_joint_pos.clone(), self.robot.data.default_joint_vel.clone()
        )
        bar_state = self.bar.data.default_root_state.clone()
        self.bar.write_root_pose_to_sim(bar_state[:, :7])
        self.bar.write_root_velocity_to_sim(bar_state[:, 7:])
        self.scene.reset()

    def set_front_camera_view(self) -> None:
        eyes = torch.tensor([FRONT_CAMERA_EYE_M], device=self.scene.device, dtype=torch.float32)
        targets = torch.tensor([FRONT_CAMERA_TARGET_M], device=self.scene.device, dtype=torch.float32)
        self.camera.set_world_poses_from_view(eyes, targets)

    def step(self, joint_targets: torch.Tensor) -> None:
        self.robot.set_joint_position_target(joint_targets)
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(self.sim.get_physics_dt())

    def image(self) -> np.ndarray:
        # Isaac Sim returns RGBA for this camera.  The ACT model was trained with RGB.
        image = self.camera.data.output["rgb"][0, ..., :3]
        return image.detach().cpu().numpy().astype(np.uint8, copy=False)

"""Nominal Isaac Lab scene for the real SO-101 bar-to-basket task."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import Camera, CameraCfg, ContactSensor, ContactSensorCfg, TiledCamera, TiledCameraCfg
from isaaclab.utils import configclass
from isaacsim.core.utils.rotations import euler_angles_to_quat


WORKSHOP_SOURCE = Path("/home/teo/sim-to-real-so101/Sim-to-Real-SO-101-Workshop/source")
if str(WORKSHOP_SOURCE) not in sys.path:
    sys.path.insert(0, str(WORKSHOP_SOURCE))

from sim_to_real_so101.assets.so101 import S0101_CONTACT_GRASP_CFG  # noqa: E402


ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "scans"
BASKET_USD = ASSET_DIR / "basket_textured.usd"
CHOCOLATE_BAR_USD = ASSET_DIR / "chocolate_bar_textured.usd"
BASKET_ASSET_SCALE = (0.46, 0.49, 0.46)
# The scan contains extra depth around the bar; keep its horizontal width while
# compressing the source height to match the real wrapped chocolate bar.
BAR_ASSET_SCALE = (0.52, 0.20, 0.26)


# These values are the first metric approximation of the captured real frame.
# Adjust them only after comparing runtime/outputs/initial_frame.png with the real frame.
TABLE_SIZE_M = (0.72, 0.82, 0.04)
TABLE_TOP_Z_M = 0.0
# Real layout: the bar starts to the left of the arm and the basket sits back
# and to its right (as seen in outputs/captured_images/opencv__dev_video0.png).
BAR_START_POS_M = (-0.19, -0.05, 0.020)
BASKET_CENTER_M = (0.12, 0.20, 0.08)
# A deliberately reachable basket pose used only by the simple-ball controller
# validation.  The nominal bar/policy scene keeps BASKET_CENTER_M unchanged.
EASY_BALL_BASKET_CENTER_M = (0.08, 0.08, 0.08)
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


def _spawn_rigid_usd(
    path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Import a visual USD and add only the root rigid-body schema.

    The scanned bar USD has render geometry but no physics schemas.  Applying the
    root body lets Isaac Lab track it as a movable object without generating a
    potentially expensive triangle collision mesh during scene startup.
    """
    from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
    from pxr import UsdGeom, UsdPhysics

    prim = spawn_from_usd(path, cfg, translation=translation, orientation=orientation, **kwargs)
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(prim)
    mass = UsdPhysics.MassAPI.Apply(prim)
    mass.CreateMassAttr(0.018)

    # Scans have render geometry but no contact shape.  This hidden local box
    # is a low-cost collision proxy sized to the physical chocolate bar.
    stage = prim.GetStage()
    collision = UsdGeom.Cube.Define(stage, f"{prim.GetPath()}/collision_proxy")
    collision.CreateSizeAttr(1.0)
    collision.AddTranslateOp().Set((0.0, 0.0525, 0.0))
    # After the root's scan scale, this is a 2.6 cm grasp width, 1.8 cm
    # thickness and 12.2 cm length.  The visual scan includes wrapper/scan
    # depth that is wider than the physical chocolate bar and prevents the
    # SO-101 jaws from closing around it.
    collision.AddScaleOp().Set((0.05, 0.09, 0.47))
    UsdPhysics.CollisionAPI.Apply(collision.GetPrim())
    UsdGeom.Imageable(collision.GetPrim()).MakeInvisible()
    return prim


@configclass
class BarPickPlaceSceneCfg(InteractiveSceneCfg):
    """Robot, movable bar, and fixed + wrist RGB observation cameras."""

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
    # The real SO-101 is black.  The workshop USD's yellow default material is
    # useful for debugging but is far outside the policy-camera appearance.
    robot.spawn.visual_material = material((0.025, 0.025, 0.030), roughness=0.38)

    bar = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Bar",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(CHOCOLATE_BAR_USD),
            scale=BAR_ASSET_SCALE,
            func=_spawn_rigid_usd,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.018),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=BAR_START_POS_M,
            # Source axes: map source Z (the long bar axis) to task X, source
            # X to task depth, and source Y upward.
            rot=(0.5, 0.5, 0.5, 0.5),
        ),
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

    # Exact NVIDIA SO-101 workshop ego-camera configuration.  NVIDIA uses a
    # tiled camera for this gripper sensor, while the fixed external camera is
    # a regular camera embedded in the lightbox asset.
    wrist_camera = TiledCameraCfg(
        # Match NVIDIA's SO-101 workshop: this prim is part of the robot USD
        # and therefore follows the gripper without manual pose updates.
        prim_path="{ENV_REGEX_NS}/Robot/gripper/gripper_cam",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb", "depth", "instance_id_segmentation_fast"],
        colorize_instance_segmentation=True,
        spawn=sim_utils.PinholeCameraCfg(
            projection_type="pinhole",
            f_stop=100,
            focal_length=13.5,
            focus_distance=0.05,
            clipping_range=(0.01, 3.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.005, 0.06, -0.062),
            rot=euler_angles_to_quat(np.array([-45, 0, 0]), degrees=True),
            convention="opengl",
        ),
    )

    contact_grasp = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/jaw",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Bar"],
    )


@configclass
class EasyBallPickPlaceSceneCfg(BarPickPlaceSceneCfg):
    """Same calibrated workspace with a simple graspable red-ball item."""

    bar = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Bar",
        spawn=sim_utils.SphereCfg(
            radius=0.018,
            visual_material=material((0.72, 0.025, 0.025), roughness=0.38),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.012),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.19, -0.05, 0.019)),
    )


@configclass
class EasyCubePickPlaceSceneCfg(BarPickPlaceSceneCfg):
    """Calibrated workspace with a flat-sided object for physical grasp validation."""

    bar = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Bar",
        spawn=sim_utils.CuboidCfg(
            size=(0.030, 0.030, 0.050),
            visual_material=material((0.04, 0.22, 0.75), roughness=0.55),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                angular_damping=10.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.012),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.19, -0.05, 0.026)),
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


def _spawn_static_usd(
    path: str,
    usd_path: Path,
    translation: tuple[float, float, float],
    *,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    orientation: tuple[float, float, float, float] | None = None,
    visual_material: sim_utils.PreviewSurfaceCfg | None = None,
) -> None:
    """Add a scanned visual asset without creating another dynamic body."""
    cfg = sim_utils.UsdFileCfg(
        usd_path=str(usd_path),
        scale=scale,
        visual_material=visual_material,
    )
    cfg.func(path, cfg, translation=translation, orientation=orientation)


def _spawn_invisible_collision_box(
    path: str,
    size: tuple[float, float, float],
    translation: tuple[float, float, float],
) -> None:
    """Create a static invisible primitive collider for a scanned asset."""
    from isaaclab.sim.utils import get_current_stage
    from pxr import UsdGeom, UsdPhysics

    cube = UsdGeom.Cube.Define(get_current_stage(), path)
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(translation)
    cube.AddScaleOp().Set(size)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdGeom.Imageable(cube.GetPrim()).MakeInvisible()


def spawn_static_workspace(*, basket_center: tuple[float, float, float] = BASKET_CENTER_M) -> None:
    """Build the desk, three-sided white enclosure, and scanned basket."""
    _spawn_static_cuboid(
        "/World/Table",
        TABLE_SIZE_M,
        (0.08, 0.08, TABLE_TOP_Z_M - TABLE_SIZE_M[2] / 2),
        # Warm wood tone to match the real tabletop better than the earlier white base.
        (0.64, 0.47, 0.28),
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
        (-0.45, 0.08, 0.365),
        (0.90, 0.91, 0.90),
        collision=False,
    )
    _spawn_static_cuboid(
        "/World/RightWall",
        (0.03, 1.40, 0.75),
        (0.60, 0.08, 0.365),
        (0.90, 0.91, 0.90),
        collision=False,
    )

    # Match the cool overhead illumination of the physical white enclosure.
    dome = sim_utils.DomeLightCfg(
        intensity=900.0,
        enable_color_temperature=True,
        color_temperature=6000.0,
    )
    dome.func("/World/FillLight", dome)
    key = sim_utils.DiskLightCfg(
        intensity=450.0,
        radius=0.22,
        enable_color_temperature=True,
        color_temperature=4800.0,
    )
    key.func("/World/KeyLight", key, translation=(0.0, -0.10, 0.75))

    # The scan's Y axis is vertical.  The USD was converted from the textured
    # GLB with embedded material assets, so keep its original appearance.
    _spawn_static_usd(
        "/World/Basket",
        BASKET_USD,
        (basket_center[0], basket_center[1], 0.0),
        scale=BASKET_ASSET_SCALE,
        # Scan coordinates are Y-up; Isaac is Z-up.  Map the scan's vertical
        # Y axis to Isaac Z so the basket stands upright on the table.
        orientation=(0.7071068, 0.7071068, 0.0, 0.0),
    )

    # Catch-volume proxy for a meaningful drop test.  It stays invisible; the
    # textured scan remains the visible basket.
    bx, by, _ = basket_center
    _spawn_invisible_collision_box("/World/BasketCollision/Base", (0.20, 0.17, 0.018), (bx, by, 0.009))
    _spawn_invisible_collision_box("/World/BasketCollision/Left", (0.018, 0.17, 0.10), (bx - 0.10, by, 0.055))
    _spawn_invisible_collision_box("/World/BasketCollision/Right", (0.018, 0.17, 0.10), (bx + 0.10, by, 0.055))
    _spawn_invisible_collision_box("/World/BasketCollision/Front", (0.20, 0.018, 0.10), (bx, by - 0.085, 0.055))
    _spawn_invisible_collision_box("/World/BasketCollision/Back", (0.20, 0.018, 0.10), (bx, by + 0.085, 0.055))


class BarPickPlaceScene:
    """Thin wrapper around Isaac Lab scene objects used by the evaluator."""

    def __init__(
        self,
        sim: sim_utils.SimulationContext,
        *,
        item: str = "bar",
        basket_center: tuple[float, float, float] | None = None,
    ) -> None:
        if item not in {"bar", "ball", "cube"}:
            raise ValueError(f"Unsupported item: {item}")
        self.basket_center = basket_center or (
            EASY_BALL_BASKET_CENTER_M if item in {"ball", "cube"} else BASKET_CENTER_M
        )
        spawn_static_workspace(basket_center=self.basket_center)
        if item == "ball":
            cfg = EasyBallPickPlaceSceneCfg()
        elif item == "cube":
            cfg = EasyCubePickPlaceSceneCfg()
        else:
            cfg = BarPickPlaceSceneCfg()
        self.scene = InteractiveScene(cfg)
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

    @property
    def wrist_camera(self) -> TiledCamera:
        return self.scene["wrist_camera"]

    @property
    def contact_grasp(self) -> ContactSensor:
        return self.scene["contact_grasp"]

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

    def wrist_image(self) -> np.ndarray:
        image = self.wrist_camera.data.output["rgb"][0, ..., :3]
        return image.detach().cpu().numpy().astype(np.uint8, copy=False)

    def images(self) -> dict[str, np.ndarray]:
        return {"front": self.image(), "wrist": self.wrist_image()}

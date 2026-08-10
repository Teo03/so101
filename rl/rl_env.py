"""State-based PPO task for the SO-101 bar-to-basket workspace.

This is intentionally a *teacher* environment: PPO receives privileged object
state so we can first validate contact physics and the reward.  Successful
rollouts can subsequently be rendered from the calibrated policy camera and
used to train an RGB ACT policy for real-world transfer.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg, FrameTransformer, FrameTransformerCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

from isaac.scene import (
    BAR_ASSET_SCALE,
    BAR_START_POS_M,
    BASKET_ASSET_SCALE,
    BASKET_CENTER_M,
    BASKET_USD,
    CHOCOLATE_BAR_USD,
    REAL_RESET_JOINTS_RAD,
    _spawn_rigid_usd,
    _spawn_invisible_collision_box,
    _spawn_static_cuboid,
    _spawn_static_usd,
    material,
)

from sim_to_real_so101.assets.so101 import S0101_CONTACT_GRASP_CFG


# The target is the *interior* of the scanned basket's collision proxy.
# The bar must be released inside this volume to count as success.
BUCKET_INNER_HALF_EXTENTS_M = (0.075, 0.060)
BUCKET_SUCCESS_Z_RANGE_M = (0.014, 0.090)


@configclass
class SO101BarBucketEnvCfg(DirectRLEnvCfg):
    """Configuration for vectorized state-based RL training."""

    decimation = 4  # 30 Hz control, 120 Hz physics -- same control rate as ACT.
    episode_length_s = 10.0
    action_space = 6
    # q, dq, end-effector/bar/bucket relative positions, bar velocity, bar orientation, last action
    observation_space = 32
    state_space = 0

    sim = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.15,
            dynamic_friction=0.95,
            restitution=0.0,
        ),
    )
    scene = InteractiveSceneCfg(
        num_envs=64,
        env_spacing=1.25,
        # Scanned USD references need regular USD cloning. Fabric cloning is
        # faster but leaves the scan's root rigid body uninitialized in env>0.
        replicate_physics=False,
        clone_in_fabric=False,
    )

    robot_cfg: ArticulationCfg = S0101_CONTACT_GRASP_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot_cfg.init_state.pos = (0.05, -0.25, 0.0)
    robot_cfg.init_state.rot = (0.0, 0.0, 0.0, 1.0)
    robot_cfg.init_state.joint_pos = REAL_RESET_JOINTS_RAD
    robot_cfg.spawn.visual_material = material((0.025, 0.025, 0.030), roughness=0.38)

    bar_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Bar",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(CHOCOLATE_BAR_USD),
            scale=BAR_ASSET_SCALE,
            func=_spawn_rigid_usd,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.018),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=BAR_START_POS_M, rot=(0.5, 0.5, 0.5, 0.5)),
    )

    ee_frame_cfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        debug_vis=False,
        target_frames=[FrameTransformerCfg.FrameCfg(prim_path="/World/envs/env_.*/Robot/gripper", name="gripper")],
    )
    # This follows the SO-101 workshop's vial task: the moving jaw is filtered
    # to the manipulation object, so reward cannot be earned by merely hovering.
    contact_grasp_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/jaw",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=["/World/envs/env_.*/Bar"],
    )

    # Incremental joint-position action scales in rad/s.  The final Jaw scale
    # is smaller to prevent the policy from slamming the low-cost gripper.
    action_speed_scales = (1.4, 1.4, 1.4, 1.7, 1.7, 0.75)
    # Deliberately randomize only the pickup object.  This covers a useful
    # portion of the real left-side pickup area while keeping the calibrated
    # basket, robot pose, and policy-camera geometry fixed.
    bar_reset_xy_noise = (0.050, 0.050)

    # Reward weights.  Grasp and lift are one-time events; this removes the
    # hovering exploit from the initial prototype reward.
    reward_reach = 1.0
    reward_grasp_event = 3.0
    reward_lift_event = 6.0
    reward_carry_to_bucket = 4.0
    reward_success = 30.0
    penalty_action = 0.002
    penalty_time = 0.01
    grasp_force_threshold = 0.10


class SO101BarBucketEnv(DirectRLEnv):
    """PPO teacher task: reach, close, lift, transport, and release the bar."""

    cfg: SO101BarBucketEnvCfg

    def __init__(self, cfg: SO101BarBucketEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.control_dt = self.cfg.sim.dt * self.cfg.decimation
        self._action_scale = torch.tensor(self.cfg.action_speed_scales, device=self.device)
        self._joint_lower = self._robot.data.soft_joint_pos_limits[0, :, 0]
        self._joint_upper = self._robot.data.soft_joint_pos_limits[0, :, 1]
        self._joint_targets = self._robot.data.default_joint_pos.clone()
        self._actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._ever_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._ever_lifted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _setup_scene(self) -> None:
        self._robot = Articulation(self.cfg.robot_cfg)
        self._bar = RigidObject(self.cfg.bar_cfg)
        self._ee_frame = FrameTransformer(self.cfg.ee_frame_cfg)
        self._contact_grasp = ContactSensor(self.cfg.contact_grasp_cfg)

        # All static visuals/colliders are created in env_0 then cloned along
        # with the robot and bar.  The scanned basket stays visible while its
        # calibrated low-cost proxy gives PPO a real catch volume to interact with.
        _spawn_static_cuboid(
            "/World/envs/env_0/Table",
            (0.72, 0.82, 0.04),
            (0.08, 0.08, -0.02),
            (0.64, 0.47, 0.28),
        )
        _spawn_static_usd(
            "/World/envs/env_0/Basket",
            BASKET_USD,
            (BASKET_CENTER_M[0], BASKET_CENTER_M[1], 0.0),
            scale=BASKET_ASSET_SCALE,
            orientation=(0.7071068, 0.7071068, 0.0, 0.0),
        )
        bx, by, _ = BASKET_CENTER_M
        basket_root = "/World/envs/env_0/BasketCollision"
        _spawn_invisible_collision_box(f"{basket_root}/Base", (0.20, 0.17, 0.018), (bx, by, 0.009))
        _spawn_invisible_collision_box(f"{basket_root}/Left", (0.018, 0.17, 0.10), (bx - 0.10, by, 0.055))
        _spawn_invisible_collision_box(f"{basket_root}/Right", (0.018, 0.17, 0.10), (bx + 0.10, by, 0.055))
        _spawn_invisible_collision_box(f"{basket_root}/Front", (0.20, 0.018, 0.10), (bx, by - 0.085, 0.055))
        _spawn_invisible_collision_box(f"{basket_root}/Back", (0.20, 0.018, 0.10), (bx, by + 0.085, 0.055))
        self.scene.clone_environments(copy_from_source=True)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["bar"] = self._bar
        self.scene.sensors["ee_frame"] = self._ee_frame
        self.scene.sensors["contact_grasp"] = self._contact_grasp
        dome = sim_utils.DomeLightCfg(intensity=900.0, color=(0.92, 0.94, 1.0))
        dome.func("/World/RLFillLight", dome)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = torch.clamp(actions, -1.0, 1.0)
        self._joint_targets += self._actions * self._action_scale * self.control_dt
        self._joint_targets = torch.clamp(self._joint_targets, self._joint_lower, self._joint_upper)

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self._joint_targets)

    def _task_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        origins = self.scene.env_origins
        ee_pos = self._ee_frame.data.target_pos_w[:, 0, :] - origins
        bar_pos = self._bar.data.root_pos_w - origins
        bar_vel = self._bar.data.root_lin_vel_w
        bar_quat = self._bar.data.root_quat_w
        bucket = torch.tensor(BASKET_CENTER_M, device=self.device).expand(self.num_envs, -1)
        return ee_pos, bar_pos, bar_vel, bar_quat, bucket

    def _bar_grasped(self, ee_pos: torch.Tensor, bar_pos: torch.Tensor) -> torch.Tensor:
        """Require jaw-to-bar contact, proximity, and a closed SO-101 jaw."""
        force = torch.linalg.vector_norm(self._contact_grasp.data.net_forces_w, dim=-1).sum(dim=1)
        close_enough = torch.linalg.vector_norm(ee_pos - bar_pos, dim=-1) < 0.055
        jaw_closed = self._robot.data.joint_pos[:, 5] < 0.0
        return (force > self.cfg.grasp_force_threshold) & close_enough & jaw_closed

    def _get_observations(self) -> dict:
        ee_pos, bar_pos, bar_vel, bar_quat, bucket = self._task_state()
        span = torch.clamp(self._joint_upper - self._joint_lower, min=1e-4)
        joint_pos = 2.0 * (self._robot.data.joint_pos - self._joint_lower) / span - 1.0
        grasped = self._bar_grasped(ee_pos, bar_pos).float().unsqueeze(-1)
        obs = torch.cat(
            (
                joint_pos,
                0.1 * self._robot.data.joint_vel,
                ee_pos - bar_pos,
                bar_pos - bucket,
                bar_vel,
                bar_quat,
                self._actions,
                grasped,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        ee_pos, bar_pos, _, _, bucket = self._task_state()
        distance = torch.linalg.vector_norm(ee_pos - bar_pos, dim=-1)
        xy_to_bucket = torch.linalg.vector_norm((bar_pos - bucket)[:, :2], dim=-1)
        bar_height = bar_pos[:, 2]
        jaw_open = self._robot.data.joint_pos[:, 5] > 0.90
        grasped = self._bar_grasped(ee_pos, bar_pos)
        lifted = bar_height > 0.055
        grasp_event = grasped & (~self._ever_grasped)
        self._ever_grasped |= grasped
        lift_event = lifted & self._ever_grasped & (~self._ever_lifted)
        self._ever_lifted |= lifted & self._ever_grasped
        inside_bucket = (
            ((bar_pos[:, 0] - bucket[:, 0]).abs() < BUCKET_INNER_HALF_EXTENTS_M[0])
            & ((bar_pos[:, 1] - bucket[:, 1]).abs() < BUCKET_INNER_HALF_EXTENTS_M[1])
            & (bar_height > BUCKET_SUCCESS_Z_RANGE_M[0])
            & (bar_height < BUCKET_SUCCESS_Z_RANGE_M[1])
        )
        released = ~grasped
        settled = torch.linalg.vector_norm(self._bar.data.root_lin_vel_w, dim=-1) < 0.15
        success = inside_bucket & jaw_open & released & settled & self._ever_lifted
        self._last_success = success

        # Stage-gated shaping: reaching is disabled after real contact; carrying
        # pays only while the bar is physically off the table after a verified grasp.
        reach = torch.exp(-distance / 0.045) * (~self._ever_grasped).float()
        carry = torch.exp(-xy_to_bucket / 0.10) * lifted.float() * self._ever_grasped.float()
        action_cost = torch.sum(torch.square(self._actions), dim=-1)
        return (
            self.cfg.reward_reach * reach
            + self.cfg.reward_grasp_event * grasp_event.float()
            + self.cfg.reward_lift_event * lift_event.float()
            + self.cfg.reward_carry_to_bucket * carry
            + self.cfg.reward_success * success.float()
            - self.cfg.penalty_action * action_cost
            - self.cfg.penalty_time
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        _, bar_pos, _, _, _ = self._task_state()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        dropped = bar_pos[:, 2] < -0.025
        out_of_workspace = (bar_pos[:, 0].abs() > 0.55) | (bar_pos[:, 1].abs() > 0.55)
        return self._last_success | dropped | out_of_workspace, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        origins = self.scene.env_origins[env_ids]

        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += origins
        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._joint_targets[env_ids] = joint_pos

        bar_state = self._bar.data.default_root_state[env_ids].clone()
        bar_state[:, :3] += origins
        bar_state[:, 0] += sample_uniform(-self.cfg.bar_reset_xy_noise[0], self.cfg.bar_reset_xy_noise[0], (len(env_ids),), self.device)
        bar_state[:, 1] += sample_uniform(-self.cfg.bar_reset_xy_noise[1], self.cfg.bar_reset_xy_noise[1], (len(env_ids),), self.device)
        bar_state[:, 7:] = 0.0
        self._bar.write_root_pose_to_sim(bar_state[:, :7], env_ids)
        self._bar.write_root_velocity_to_sim(bar_state[:, 7:], env_ids)
        self._ever_lifted[env_ids] = False
        self._ever_grasped[env_ids] = False
        self._last_success[env_ids] = False

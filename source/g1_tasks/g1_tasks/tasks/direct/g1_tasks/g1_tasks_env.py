# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform, quat_apply_inverse

from .g1_tasks_env_cfg import G1TasksEnvCfg


class G1TasksEnv(DirectRLEnv):
    """Environment for training Unitree G1 humanoid to walk."""
    cfg: G1TasksEnvCfg

    def __init__(self, cfg: G1TasksEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Define which joints to control (exclude hands for walking task)
        self._controlled_joints = [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
        ]

        # Get indices of controlled joints
        self._controlled_joint_indices = []
        for joint_name in self._controlled_joints:
            idx, _ = self.robot.find_joints(joint_name)
            if len(idx) > 0:
                self._controlled_joint_indices.append(idx[0])

        self._controlled_joint_indices = torch.tensor(self._controlled_joint_indices, device=self.device)

        # Get body index
        body_names = self.robot.body_names
        if "pelvis" in body_names:
            self._body_id = self.robot.find_bodies("pelvis")[0][0]
        elif "base_link" in body_names:
            self._body_id = self.robot.find_bodies("base_link")[0][0]
        else:
            self._body_id = 0  # Use root body

        # Store velocities and state
        self.base_lin_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.base_ang_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.previous_actions = torch.zeros(self.num_envs, len(self._controlled_joints), device=self.device)
        self.projected_gravity = torch.zeros(self.num_envs, 3, device=self.device)

        # Command: target velocity
        self.commands = torch.zeros(self.num_envs, 3, device=self.device)
        self.commands[:, 0] = self.cfg.target_velocity  # forward velocity

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        # Apply actions only to controlled joints
        # Create full joint target (all joints at default)
        joint_targets = self.robot.data.default_joint_pos.clone()

        # Update only the controlled joints
        joint_targets[:, self._controlled_joint_indices] += self.actions * self.cfg.action_scale

        # Apply the targets
        self.robot.set_joint_position_target(joint_targets)

    def _get_observations(self) -> dict:
        # Update base velocity and projected gravity
        self._compute_intermediate_values()

        # Get joint positions and velocities for controlled joints only
        dof_pos = (self.robot.data.joint_pos[:, self._controlled_joint_indices] - self.robot.data.default_joint_pos[:, self._controlled_joint_indices])
        dof_vel = self.robot.data.joint_vel[:, self._controlled_joint_indices]

        obs = torch.cat(
            (
                self.base_lin_vel,  # 3
                self.base_ang_vel,  # 3
                self.projected_gravity,  # 3
                self.commands,  # 3
                dof_pos,  # num_controlled_dof
                dof_vel,  # num_controlled_dof
                self.actions,  # num_controlled_dof
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
            self.cfg.rew_scale_lin_vel_xy,
            self.cfg.rew_scale_ang_vel_z,
            self.cfg.rew_scale_joint_torques,
            self.cfg.rew_scale_joint_acc,
            self.cfg.rew_scale_action_rate,
            self.cfg.rew_scale_orientation,
            self.cfg.rew_scale_base_height,
            self.cfg.rew_scale_dof_vel,
            self.cfg.rew_scale_dof_pos_limits,
            self.cfg.rew_scale_termination,
            self.base_lin_vel,
            self.base_ang_vel,
            self.robot.data.joint_vel[:, self._controlled_joint_indices],
            self.robot.data.applied_torque[:, self._controlled_joint_indices],
            self.commands[:, :2],
            self.actions,
            self.previous_actions,
            self.projected_gravity,
            self.robot.data.root_pos_w[:, 2],
            self.robot.data.joint_pos[:, self._controlled_joint_indices],
            self.robot.data.soft_joint_pos_limits[:, self._controlled_joint_indices, :],
            self.reset_terminated,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
 
        # Check if robot has fallen (base too low or tilted)
        base_height = self.robot.data.root_pos_w[:, 2]
        has_fallen = base_height < self.cfg.termination_height

        # Check if robot is too tilted
        is_tilted = torch.abs(self.projected_gravity[:, 2]) < 0.5  # z-component of up vector

        terminated = has_fallen | is_tilted
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # Reset joint states with small random noise
        joint_pos = self.robot.data.default_joint_pos[env_ids]

        # Add noise only to controlled joints
        noise = torch.zeros_like(joint_pos)
        noise[:, self._controlled_joint_indices] = sample_uniform(
            -0.05, 0.05, (len(env_ids), len(self._controlled_joint_indices)), joint_pos.device
        )
        joint_pos += noise

        joint_vel = torch.zeros_like(joint_pos)

        # Reset root state
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # Add small random height variation
        default_root_state[:, 2] += sample_uniform(
            -0.02, 0.02, (len(env_ids),), default_root_state.device
        )

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Reset buffers
        self.base_lin_vel[env_ids] = 0.0
        self.base_ang_vel[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0

    def _compute_intermediate_values(self):
        # Compute base linear velocity in base frame
        self.base_lin_vel[:] = quat_apply_inverse(
            self.robot.data.root_quat_w, self.robot.data.root_lin_vel_w
        )

        # Compute base angular velocity in base frame
        self.base_ang_vel[:] = quat_apply_inverse(
            self.robot.data.root_quat_w, self.robot.data.root_ang_vel_w
        )

        # Compute projected gravity (gravity vector in base frame)
        self.projected_gravity[:] = quat_apply_inverse(
            self.robot.data.root_quat_w,
            torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1),
        )


@torch.jit.script
def compute_rewards(
    rew_scale_lin_vel_xy: float,
    rew_scale_ang_vel_z: float,
    rew_scale_joint_torques: float,
    rew_scale_joint_acc: float,
    rew_scale_action_rate: float,
    rew_scale_orientation: float,
    rew_scale_base_height: float,
    rew_scale_dof_vel: float,
    rew_scale_dof_pos_limits: float,
    rew_scale_termination: float,
    base_lin_vel: torch.Tensor,
    base_ang_vel: torch.Tensor,
    dof_vel: torch.Tensor,
    dof_torques: torch.Tensor,
    commands: torch.Tensor,
    actions: torch.Tensor,
    previous_actions: torch.Tensor,
    projected_gravity: torch.Tensor,
    base_height: torch.Tensor,
    dof_pos: torch.Tensor,
    dof_pos_limits: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    lin_vel_error = torch.sum(torch.square(commands - base_lin_vel[:, :2]), dim=1)
    rew_lin_vel_xy = rew_scale_lin_vel_xy * torch.exp(-lin_vel_error / 0.25)

    rew_ang_vel_z = -rew_scale_ang_vel_z * torch.square(base_ang_vel[:, 2])
    rew_joint_torques = -rew_scale_joint_torques * torch.sum(torch.square(dof_torques), dim=1)
    rew_joint_acc = -rew_scale_joint_acc * torch.sum(torch.square(dof_vel), dim=1)

    rew_action_rate = -rew_scale_action_rate * torch.sum(
        torch.square(actions - previous_actions), dim=1
    )

    rew_orientation = -rew_scale_orientation * torch.square(projected_gravity[:, 2] - 1.0)
    rew_base_height = -rew_scale_base_height * torch.square(base_height - 0.85)

    rew_dof_vel = -rew_scale_dof_vel * torch.sum(torch.square(dof_vel), dim=1)

    dof_pos_normalized = (dof_pos - dof_pos_limits[:, :, 0]) / (
        dof_pos_limits[:, :, 1] - dof_pos_limits[:, :, 0]
    )
    out_of_limits = -(dof_pos_normalized - 0.5).abs() + 0.5
    rew_dof_pos_limits = -rew_scale_dof_pos_limits * torch.sum(
        torch.clamp(out_of_limits, min=0.0), dim=1
    )

    rew_termination = -rew_scale_termination * reset_terminated.float()

    return (
        rew_lin_vel_xy
        + rew_ang_vel_z
        + rew_joint_torques
        + rew_joint_acc
        + rew_action_rate
        + rew_orientation
        + rew_base_height
        + rew_dof_vel
        + rew_dof_pos_limits
        + rew_termination
    )
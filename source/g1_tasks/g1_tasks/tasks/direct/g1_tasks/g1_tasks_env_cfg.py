# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg

import isaaclab.sim as sim_utils


@configclass
class G1TasksEnvCfg(DirectRLEnvCfg):
    """Configuration for the Unitree G1 walking environment."""
    
    # Environment settings
    decimation = 4
    episode_length_s = 20.0
    
    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,  # 200 Hz simulation
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # Robot configuration - Unitree G1 humanoid
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/roshai/Documents/projects/RL/model_assets/G1/g1.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.8),  # Initial height for G1
            joint_pos={
                # Left leg
                "left_hip_pitch_joint": 0.0,
                "left_hip_roll_joint": 0.0,
                "left_hip_yaw_joint": 0.0,
                "left_knee_joint": 0.0,
                "left_ankle_pitch_joint": 0.0,
                "left_ankle_roll_joint": 0.0,
                # Right leg
                "right_hip_pitch_joint": 0.0,
                "right_hip_roll_joint": 0.0,
                "right_hip_yaw_joint": 0.0,
                "right_knee_joint": 0.0,
                "right_ankle_pitch_joint": 0.0,
                "right_ankle_roll_joint": 0.0,
                # Waist/Torso (correct joint names)
                "waist_yaw_joint": 0.0,
                "waist_roll_joint": 0.0,
                "waist_pitch_joint": 0.0,
                # Left arm
                "left_shoulder_pitch_joint": 0.0,
                "left_shoulder_roll_joint": 0.0,
                "left_shoulder_yaw_joint": 0.0,
                "left_elbow_joint": 0.0,
                "left_wrist_roll_joint": 0.0,
                "left_wrist_pitch_joint": 0.,
                "left_wrist_yaw_joint": 0.0,
                # Right arm
                "right_shoulder_pitch_joint": 0.0,
                "right_shoulder_roll_joint": 0.0,
                "right_shoulder_yaw_joint": 0.0,
                "right_elbow_joint": 0.0,
                "right_wrist_roll_joint": 0.0,
                "right_wrist_pitch_joint": 0.0,
                "right_wrist_yaw_joint": 0.0,
                # Hands - set to small values to keep fingers slightly closed
                "left_hand_index_0_joint": 0.0,
                "left_hand_middle_0_joint": 0.0,
                "left_hand_thumb_0_joint": 0.0,
                "right_hand_index_0_joint": 0.0,
                "right_hand_middle_0_joint": 0.0,
                "right_hand_thumb_0_joint": 0.0,
                "left_hand_index_1_joint": 0.0,
                "left_hand_middle_1_joint": 0.0,
                "left_hand_thumb_1_joint": 0.0,
                "right_hand_index_1_joint": 0.0,
                "right_hand_middle_1_joint": 0.0,
                "right_hand_thumb_1_joint": 0.0,
                "left_hand_thumb_2_joint": 0.0,
                "right_hand_thumb_2_joint": 0.0,
            },
        ),
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_.*", ".*_knee.*", ".*_ankle_.*"],
                stiffness=80.0,
                damping=4.0,
            ),
            "torso": ImplicitActuatorCfg(
                joint_names_expr=["waist_.*"],
                stiffness=80.0,
                damping=4.0,
            ),
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[".*_shoulder_.*", ".*_elbow.*", ".*_wrist_.*"],
                stiffness=40.0,
                damping=2.0,
            ),
            "hands": ImplicitActuatorCfg(
                joint_names_expr=[".*_hand_.*"],
                stiffness=20.0,
                damping=1.0,
            ),
        },
    )

    # Scene configuration
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=6000,
        env_spacing=4.0,
        replicate_physics=True
    )

    # Action and observation spaces
    # G1 has 37 DOF total (legs: 12, waist: 3, arms: 14, hands: 8)
    # For walking, we'll control legs + waist + arms (29 DOF), keep hands fixed
    action_space = 29  # 12 (legs) + 3 (waist) + 14 (arms)
    # Observation: base_vel(3) + ang_vel(3) + gravity(3) + commands(3) + dof_pos(29) + dof_vel(29) + actions(29)
    observation_space = 99
    state_space = 0

    # Action parameters
    action_scale = 0.5  # Maximum joint angle deviation in radians

    # Command parameters
    target_velocity = 1.0  # Target forward velocity in m/s

    # Reward scales
    rew_scale_lin_vel_xy = 2.0          # Reward for tracking velocity
    rew_scale_ang_vel_z = -0.05         # Penalty for spinning
    rew_scale_joint_torques = -2e-5     # Penalty for high torques
    rew_scale_joint_acc = -2.5e-7           # Penalty for high accelerations
    rew_scale_action_rate = -0.01       # Penalty for rapid action changes
    rew_scale_orientation = -5.0        # Penalty for tilting
    rew_scale_base_height = -1.0        # Penalty for wrong height
    rew_scale_dof_vel = -5e-4           # Penalty for high joint velocities
    rew_scale_dof_pos_limits = -10.0    # Penalty for approaching joint limits
    rew_scale_termination = -2.0        # Penalty for falling

    # Termination conditions
    termination_height = 0.8  # Terminate if base drops below this height (meters)


# Optional: Create a configuration for standing still (useful for initial training)
@configclass
class G1StandEnvCfg(G1TasksEnvCfg):
    """Configuration for G1 standing balance task."""
    
    episode_length_s = 10.0
    target_velocity = 0.0  # No forward velocity, just balance
    
    # Adjust reward scales for standing
    rew_scale_lin_vel_xy = -1.0  # Penalize movement
    rew_scale_base_height = -5.0  # Strong penalty for height deviation
    rew_scale_orientation = -10.0  # Strong penalty for tilting


# Optional: Create a configuration for slow walking (curriculum learning)
@configclass
class G1SlowWalkEnvCfg(G1TasksEnvCfg):
    """Configuration for G1 slow walking task."""
    
    target_velocity = 0.3  # Slower target velocity
    
    # More lenient penalties for initial learning
    rew_scale_joint_torques = -1e-5
    rew_scale_joint_acc = -1e-7
    rew_scale_action_rate = -0.005


# Optional: Simplified configuration that only controls legs (easier to learn)
@configclass
class G1WalkLegsOnlyEnvCfg(G1TasksEnvCfg):
    """Configuration for G1 walking with only leg control."""
    
    # Only control legs, keep arms and waist at default positions
    action_space = 12  # Only leg joints
    observation_space = 51  # base_vel(3) + ang_vel(3) + gravity(3) + commands(3) + dof_pos(12) + dof_vel(12) + actions(12)
    
    target_velocity = 0.5  # Start slower
    
    # More lenient penalties
    rew_scale_joint_torques = -1e-5
    rew_scale_joint_acc = -1e-7
    rew_scale_action_rate = -0.005
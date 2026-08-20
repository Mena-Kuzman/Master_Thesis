from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class RLTrainArgs:
    """Centralized defaults for SAC training + env wiring."""

    total_timesteps: int = 1_000_000
    seed: int = 1    #random seed for reproducibility
    cuda: bool = True
    torch_deterministic: bool = True   #if toggled, `torch.backends.cudnn.deterministic=False

    buffer_size: int = 1_000_000
    gamma: float = 0.99 #the discount factor gamma
    tau: float = 0.005  #target smoothing coefficient (default: 0.005)
    batch_size: int = 2048
    learning_starts: int = 6_000        #sum for all actors, not per-actor
    policy_lr: float = 3e-4     #the learning rate of the policy network optimizer
    q_lr: float = 1e-3      #the learning rate of the Q network optimizer
    policy_frequency: int = 2       #the frequency of training policy (delayed)
    target_network_frequency: int = 1       #the frequency of updates for the target networks
    log_interval: int = 2000
    episode_print_interval: int = 1
    alpha: float = 0.2      #Entropy regularization coefficient
    autotune: bool = False

    # Parallelization
    num_envs: int = 1

    # Env setup
    headless: bool = True
    frame_skip: int = 10
    episode_length: int = 4000
    command_forward: float = 0.35
    command_lateral: float = 0.0
    command_yaw: float = 0.0
    command_pitch: float = 0.0
    stair_start_x: float = 0.5
    residual_warmup_steps: int = 20
    residual_ramp_steps: int = 40
    overspeed_terminate_mps: float = 1.0
    hard_attitude_terminate_rad: float = 1.5
    hard_attitude_violation_steps: int = 50
    base_height_terminate: float = 0.10
    base_height_grace: float = 0.06
    command_ramp_steps: int = 80
    startup_max_cmd_mps: float = 0.25
    startup_torque_scale_steps: int = 30
    startup_torque_scale: float = 0.6
    train_swing_time: bool = True

    # Environment selection
    terrain_type: str = "procedural"  # "stairs", "procedural", or "procedural_narrow"
    survival_success_duration_steps: int = 3500
    six_stage_curriculum_enabled: bool = True

    # Procedural terrain sizing (meters)
    procedural_terrain_length_m: float = 15.0
    procedural_terrain_width_m: float = 15.0
    procedural_narrow_terrain_length_m: float = 20.0
    procedural_narrow_terrain_width_m: float = 3.0
    procedural_narrow_success_x: float = 10.0

    # Six-stage curriculum
    curriculum_max_stage: int = 6
    curriculum_stage_1_min: float = 0.3
    curriculum_stage_1_max: float = 0.4
    curriculum_stage_2_min: float = 0.4
    curriculum_stage_2_max: float = 0.5
    curriculum_stage_3_min: float = 0.5
    curriculum_stage_3_max: float = 0.6
    curriculum_stage_4_min: float = 0.6
    curriculum_stage_4_max: float = 0.7
    curriculum_stage_5_min: float = 0.7
    curriculum_stage_5_max: float = 0.8
    curriculum_stage_6_min: float = 0.8
    curriculum_stage_6_max: float = 1.0
    curriculum_early_stair_start_x_min: float = 1.5
    curriculum_early_stair_start_x_max: float = 1.5
    curriculum_early_stair_height_min: float = 0.10
    curriculum_early_stair_height_max: float = 0.12
    curriculum_late_stair_start_x_min: float = 2.0
    curriculum_late_stair_start_x_max: float = 2.0
    curriculum_late_stair_height_min: float = 0.13
    curriculum_late_stair_height_max: float = 0.15
    curriculum_early_yaw_min: float = 0.0
    curriculum_early_yaw_max: float = 0.0
    curriculum_late_yaw_min: float = 0.0
    curriculum_late_yaw_max: float = 0.0
    curriculum_success_threshold: float = 0.70
    # Shared window used by curriculum updates and rolling success-rate logging.
    curriculum_window_episodes: int = 10
    curriculum_min_stage_episodes: int = 20

    # Legacy/optional modes
    speed_curriculum_enabled: bool = False
    speed_curriculum_min: float = 0.35
    speed_curriculum_max: float = 0.8
    speed_curriculum_warmup_episodes: int = 5
    speed_curriculum_full_episodes: int = 35
    random_command_enabled: bool = False
    random_command_min: float = 0.4
    random_command_max: float = 0.7

    # Stair randomization
    randomize_stairs: bool = False
    fixed_course: bool = False
    stair_start_x_min: float = 1.5
    stair_start_x_max: float = 3.0
    stair_step_height_min: float = 0.08
    stair_step_height_max: float = 0.15
    stair_step_width_min: float = 0.35
    stair_step_width_max: float = 0.35
    stair_num_steps_up_min: int = 3
    stair_num_steps_up_max: int = 4
    stair_num_steps_down_min: int = 3
    stair_num_steps_down_max: int = 4
    stair_equal_up_down_steps: bool = True

    # Terrain randomization between obstacle sets
    random_uneven_terrain: bool = True
    uneven_patches_per_gap_min: int = 2
    uneven_patches_per_gap_max: int = 6
    uneven_patch_height_min: float = 0.01
    uneven_patch_height_max: float = 0.05
    uneven_patch_length_min: float = 0.10
    uneven_patch_length_max: float = 0.40
    uneven_patch_width_min: float = 0.20
    uneven_patch_width_max: float = 1.20
    random_slope_terrain: bool = True
    slope_segments_per_episode_min: int = 1
    slope_segments_per_episode_max: int = 3
    slope_angle_deg_min: float = 3.0
    slope_angle_deg_max: float = 10.0
    slope_length_min: float = 0.60
    slope_length_max: float = 1.60
    slope_step_resolution_m: float = 0.12

    # Logging / checkpoints
    run_name: str = "sac_step_height_residual"
    log_dir: str = "runs"
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 20_000


@dataclass(frozen=True)
class RLParallelDefaults:
    num_actors: int = 8 #no way to use more than 12
    num_envs_per_actor: int = 1
    rollout_fragment_steps: int = 512
    learner_updates_per_batch: int = 3
    policy_broadcast_interval_updates: int = 10
    transition_queue_size: int = 128
    actor_seed_stride: int = 1000
    actor_start_stagger_sec: float = 5
    run_name: str = "sac_step_height_residual_parallel"


@dataclass(frozen=True)
class RLEnvDefaults:
    add_stairs: bool = True
    termination_grace_steps: int = 16
    attitude_violation_steps: int = 3
    overspeed_violation_steps: int = 3
    multi_obstacles_enabled: bool = True
    extra_obstacle_sets: int = 2
    obstacle_gap_min: float = 0.8
    obstacle_gap_max: float = 1.8
    random_uneven_terrain: bool = True
    uneven_patches_per_gap_min: int = 2
    uneven_patches_per_gap_max: int = 6
    uneven_patch_height_min: float = 0.01
    uneven_patch_height_max: float = 0.05
    uneven_patch_length_min: float = 0.10
    uneven_patch_length_max: float = 0.40
    uneven_patch_width_min: float = 0.20
    uneven_patch_width_max: float = 1.20
    random_slope_terrain: bool = True
    slope_segments_per_episode_min: int = 1
    slope_segments_per_episode_max: int = 3
    slope_angle_deg_min: float = 3.0
    slope_angle_deg_max: float = 10.0
    slope_length_min: float = 0.60
    slope_length_max: float = 1.60
    slope_step_resolution_m: float = 0.12
    success_reset_enabled: bool = True
    success_reset_hold_steps: int = 20
    success_reset_min_episode_step: int = 120
    success_reset_x_margin_m: float = 0.10
    post_reset_stand_steps: int = 15
    reset_recovery_base_z_min: float = 0.20
    stagnation_window_steps: int = 100
    stagnation_threshold_m: float = 0.05
    yaw_hold_penalty: float = 0.25
    course_complete_terminal_bonus: float = 0.0
    attitude_hard_terminal_penalty: float = 0.0
    stagnation_terminal_penalty: float = 0.0


RL_PARALLEL_DEFAULTS = RLParallelDefaults()
RL_ENV_DEFAULTS = RLEnvDefaults()


def build_env_kwargs(base: Any, show_pygame_ui: Optional[bool] = None) -> dict[str, Any]:
    """Create env kwargs from a training-args-like object."""
    kwargs: dict[str, Any] = {
        "headless": base.headless,
        "frame_skip": base.frame_skip,
        "episode_length": base.episode_length,
        "command_forward": base.command_forward,
        "command_lateral": base.command_lateral,
        "command_yaw": base.command_yaw,
        "command_pitch": base.command_pitch,
        "stair_start_x": base.stair_start_x,
        "curriculum_stage_1_min": base.curriculum_stage_1_min,
        "curriculum_stage_1_max": base.curriculum_stage_1_max,
        "curriculum_stage_2_min": base.curriculum_stage_2_min,
        "curriculum_stage_2_max": base.curriculum_stage_2_max,
        "curriculum_stage_3_min": base.curriculum_stage_3_min,
        "curriculum_stage_3_max": base.curriculum_stage_3_max,
        "curriculum_stage_4_min": base.curriculum_stage_4_min,
        "curriculum_stage_4_max": base.curriculum_stage_4_max,
        "curriculum_stage_5_min": base.curriculum_stage_5_min,
        "curriculum_stage_5_max": base.curriculum_stage_5_max,
        "curriculum_stage_6_min": base.curriculum_stage_6_min,
        "curriculum_stage_6_max": base.curriculum_stage_6_max,
        "curriculum_early_stair_start_x_min": base.curriculum_early_stair_start_x_min,
        "curriculum_early_stair_start_x_max": base.curriculum_early_stair_start_x_max,
        "curriculum_early_stair_height_min": base.curriculum_early_stair_height_min,
        "curriculum_early_stair_height_max": base.curriculum_early_stair_height_max,
        "curriculum_late_stair_start_x_min": base.curriculum_late_stair_start_x_min,
        "curriculum_late_stair_start_x_max": base.curriculum_late_stair_start_x_max,
        "curriculum_late_stair_height_min": base.curriculum_late_stair_height_min,
        "curriculum_late_stair_height_max": base.curriculum_late_stair_height_max,
        "curriculum_early_yaw_min": base.curriculum_early_yaw_min,
        "curriculum_early_yaw_max": base.curriculum_early_yaw_max,
        "curriculum_late_yaw_min": base.curriculum_late_yaw_min,
        "curriculum_late_yaw_max": base.curriculum_late_yaw_max,
        "curriculum_success_threshold": base.curriculum_success_threshold,
        "curriculum_window_episodes": base.curriculum_window_episodes,
        "curriculum_min_stage_episodes": base.curriculum_min_stage_episodes,
        "speed_curriculum_enabled": base.speed_curriculum_enabled,
        "speed_curriculum_min": base.speed_curriculum_min,
        "speed_curriculum_max": base.speed_curriculum_max,
        "speed_curriculum_warmup_episodes": base.speed_curriculum_warmup_episodes,
        "speed_curriculum_full_episodes": base.speed_curriculum_full_episodes,
        "random_command_enabled": base.random_command_enabled,
        "random_command_min": base.random_command_min,
        "random_command_max": base.random_command_max,
        "randomize_stairs": base.randomize_stairs,
        "stair_start_x_min": base.stair_start_x_min,
        "stair_start_x_max": base.stair_start_x_max,
        "stair_step_height_min": base.stair_step_height_min,
        "stair_step_height_max": base.stair_step_height_max,
        "stair_step_width_min": base.stair_step_width_min,
        "stair_step_width_max": base.stair_step_width_max,
        "stair_num_steps_up_min": base.stair_num_steps_up_min,
        "stair_num_steps_up_max": base.stair_num_steps_up_max,
        "stair_num_steps_down_min": base.stair_num_steps_down_min,
        "stair_num_steps_down_max": base.stair_num_steps_down_max,
        "stair_equal_up_down_steps": base.stair_equal_up_down_steps,
        "residual_warmup_steps": base.residual_warmup_steps,
        "residual_ramp_steps": base.residual_ramp_steps,
        "overspeed_terminate_mps": base.overspeed_terminate_mps,
        "hard_attitude_terminate_rad": base.hard_attitude_terminate_rad,
        "hard_attitude_violation_steps": base.hard_attitude_violation_steps,
        "base_height_terminate": base.base_height_terminate,
        "base_height_grace": base.base_height_grace,
        "command_ramp_steps": base.command_ramp_steps,
        "startup_max_cmd_mps": base.startup_max_cmd_mps,
        "startup_torque_scale_steps": base.startup_torque_scale_steps,
        "startup_torque_scale": base.startup_torque_scale,
        "train_swing_time": base.train_swing_time,
        "terrain_type": getattr(base, "terrain_type", "stairs"),
        "survival_success_duration_steps": getattr(base, "survival_success_duration_steps", 2500),
        "six_stage_curriculum_enabled": getattr(base, "six_stage_curriculum_enabled", True),
        "procedural_terrain_length_m": getattr(base, "procedural_terrain_length_m", 30.0),
        "procedural_terrain_width_m": getattr(base, "procedural_terrain_width_m", 30.0),
        "procedural_narrow_terrain_length_m": getattr(base, "procedural_narrow_terrain_length_m", 20.0),
        "procedural_narrow_terrain_width_m": getattr(base, "procedural_narrow_terrain_width_m", 5.0),
        "procedural_narrow_success_x": getattr(base, "procedural_narrow_success_x", 19.5),
        "random_uneven_terrain": base.random_uneven_terrain,
        "uneven_patches_per_gap_min": base.uneven_patches_per_gap_min,
        "uneven_patches_per_gap_max": base.uneven_patches_per_gap_max,
        "uneven_patch_height_min": base.uneven_patch_height_min,
        "uneven_patch_height_max": base.uneven_patch_height_max,
        "uneven_patch_length_min": base.uneven_patch_length_min,
        "uneven_patch_length_max": base.uneven_patch_length_max,
        "uneven_patch_width_min": base.uneven_patch_width_min,
        "uneven_patch_width_max": base.uneven_patch_width_max,
        "random_slope_terrain": base.random_slope_terrain,
        "slope_segments_per_episode_min": base.slope_segments_per_episode_min,
        "slope_segments_per_episode_max": base.slope_segments_per_episode_max,
        "slope_angle_deg_min": base.slope_angle_deg_min,
        "slope_angle_deg_max": base.slope_angle_deg_max,
        "slope_length_min": base.slope_length_min,
        "slope_length_max": base.slope_length_max,
        "slope_step_resolution_m": base.slope_step_resolution_m,
        "fixed_course": base.fixed_course,
        "learning_starts_steps": base.learning_starts,
    }
    if show_pygame_ui is not None:
        kwargs["show_pygame_ui"] = bool(show_pygame_ui)
    return kwargs

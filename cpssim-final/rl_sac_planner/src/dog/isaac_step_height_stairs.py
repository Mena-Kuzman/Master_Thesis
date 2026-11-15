from __future__ import annotations

import os
import sys
import traceback
import multiprocessing as mp
from functools import partial
from typing import Optional
from collections import deque

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as exc:
    raise ImportError("gymnasium is required for IsaacStepHeightResidualEnv") from exc


def _spawn_isaac_env(env_cls, env_kwargs):
    """Top-level factory used by multiprocessing spawn workers."""
    _ensure_casadi_binary_path()
    return env_cls(**env_kwargs)


def _ensure_casadi_binary_path() -> None:
    """Prioritize Isaac Sim site-packages for worker interpreter imports."""
    isaac_site_packages = "/home/mena/isaacsim/kit/python/lib/python3.11/site-packages"
    if os.path.isdir(isaac_site_packages) and isaac_site_packages in sys.path:
        sys.path.remove(isaac_site_packages)
        sys.path.insert(0, isaac_site_packages)


def _configure_worker_thread_limits() -> None:
    """Limit per-process math thread pools to improve multi-process scaling."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def _proc_worker(remote, env_cls, env_kwargs):
    """Worker process loop for Isaac process-vectorized environments."""
    env = None
    try:
        _configure_worker_thread_limits()
        env = _spawn_isaac_env(env_cls, env_kwargs)
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                obs, info = env.reset(seed=data)
                remote.send((obs, info, None))
            elif cmd == "step":
                obs, reward, terminated, truncated, info = env.step(data)
                if bool(terminated or truncated):
                    final_obs = obs
                    final_info = info
                    obs, reset_info = env.reset()
                    info = dict(info) if isinstance(info, dict) else {}
                    info["final_observation"] = final_obs
                    info["final_info"] = final_info
                    info["reset_info"] = reset_info
                remote.send((obs, reward, terminated, truncated, info, None))
            elif cmd == "close":
                if env is not None:
                    env.close()
                remote.send((None,))
                break
            else:
                raise RuntimeError(f"Unknown worker command: {cmd}")
    except Exception:
        remote.send((None, None, None, None, None, traceback.format_exc()))
    finally:
        try:
            if env is not None:
                env.close()
        except Exception:
            pass
        try:
            remote.close()
        except Exception:
            pass


class _IsaacProcessVectorEnv:
    """Lightweight process-based vector env compatible with Isaac singleton runtime."""

    def __init__(self, env_cls, num_envs: int, env_kwargs: dict):
        self.num_envs = int(num_envs)
        self._closed = False
        self._ctx = mp.get_context("spawn")
        self._remotes = []
        self._processes = []

        self.single_observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(env_cls.RL_OBS_DIM,),
            dtype=np.float32,
        )
        self.single_action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(env_cls.RL_ACT_DIM,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_envs, env_cls.RL_OBS_DIM),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_envs, env_cls.RL_ACT_DIM),
            dtype=np.float32,
        )

        for _ in range(self.num_envs):
            parent_remote, child_remote = self._ctx.Pipe()
            proc = self._ctx.Process(
                target=_proc_worker,
                args=(child_remote, env_cls, dict(env_kwargs)),
                daemon=False,
            )
            proc.start()
            child_remote.close()
            self._remotes.append(parent_remote)
            self._processes.append(proc)

    @staticmethod
    def _raise_if_worker_error(payload, context: str) -> None:
        """Raise a readable error when a worker reports an exception payload."""
        if isinstance(payload, tuple) and len(payload) > 0:
            worker_tb = payload[-1]
            if isinstance(worker_tb, str) and worker_tb:
                raise RuntimeError(f"Vector worker {context} failed:\n{worker_tb}")

    def reset(self, seed: Optional[int] = None):
        for i, remote in enumerate(self._remotes):
            worker_seed = (int(seed) + i) if seed is not None else None
            remote.send(("reset", worker_seed))

        obs_list = []
        info_list = []
        for remote in self._remotes:
            try:
                data = remote.recv()
            except (EOFError, ConnectionResetError, BrokenPipeError, OSError) as exc:
                raise RuntimeError(
                    "Vector worker reset failed before returning data. "
                    "Check worker startup/import logs for details."
                ) from exc
            self._raise_if_worker_error(data, context="reset")
            obs, info, _ = data
            obs_list.append(obs)
            info_list.append(info)

        return np.asarray(obs_list, dtype=np.float32), info_list

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape[0] != self.num_envs:
            raise ValueError(f"Expected action batch first dim {self.num_envs}, got {actions.shape}")

        for i, remote in enumerate(self._remotes):
            remote.send(("step", actions[i]))

        obs_list = []
        rew_list = []
        term_list = []
        trunc_list = []
        info_list = []
        for remote in self._remotes:
            try:
                data = remote.recv()
            except (EOFError, ConnectionResetError, BrokenPipeError, OSError) as exc:
                raise RuntimeError(
                    "Vector worker step failed before returning data. "
                    "Check worker logs for errors."
                ) from exc
            self._raise_if_worker_error(data, context="step")
            obs, rew, terminated, truncated, info, _ = data
            obs_list.append(obs)
            rew_list.append(rew)
            term_list.append(bool(terminated))
            trunc_list.append(bool(truncated))
            info_list.append(info)

        return (
            np.asarray(obs_list, dtype=np.float32),
            np.asarray(rew_list, dtype=np.float32),
            np.asarray(term_list, dtype=np.bool_),
            np.asarray(trunc_list, dtype=np.bool_),
            info_list,
        )

    def close(self):
        if self._closed:
            return
        self._closed = True

        for remote in self._remotes:
            try:
                remote.send(("close", None))
            except Exception:
                pass
        for remote in self._remotes:
            try:
                remote.recv()
            except Exception:
                pass
            try:
                remote.close()
            except Exception:
                pass
        for proc in self._processes:
            try:
                proc.join(timeout=10.0)
            except Exception:
                pass
            if proc.is_alive():
                try:
                    proc.terminate()
                except Exception:
                    pass


class IsaacStepHeightResidualEnv(gym.Env):
    """Gymnasium wrapper for Isaac Sim with 2D RL control over step height and swing time.

    Action:
        a[0] in [-1, 1], mapped to step-height residual meters.
        a[1] in [-1, 1], mapped to swing time in seconds within the configured range.

    Observation:
        Fixed-size vector returned by `IsaacMPCController.get_rl_observation_vector()`.
    """

    metadata = {"render_modes": ["human", "none"]}
    # Controller observation vector (88) includes residual_gate, episode_phase,
    # distance_to_course_end, and swing_time appended by IsaacMPCController.
    RL_OBS_DIM = 88
    RL_ACT_DIM = 2

    @classmethod
    def make_env(cls, num_envs: int = 1, **kwargs):
        """Create either a single env or a process-isolated AsyncVectorEnv.

        For num_envs > 1 we must isolate each environment in its own process,
        because Isaac Sim uses a singleton runtime per process.
        """
        num_envs = int(max(1, num_envs))
        if num_envs == 1:
            return cls(**kwargs)
        return _IsaacProcessVectorEnv(env_cls=cls, num_envs=num_envs, env_kwargs=kwargs)

    def __init__(
        self,
        headless: Optional[bool] = None,
        frame_skip: Optional[int] = None,
        episode_length: Optional[int] = None,
        command_forward: Optional[float] = None,
        command_lateral: Optional[float] = None,
        command_yaw: Optional[float] = None,
        command_pitch: Optional[float] = None,
        add_stairs: bool = True,
        stair_start_x: Optional[float] = None,
        residual_warmup_steps: Optional[int] = None,
        residual_ramp_steps: Optional[int] = None,
        overspeed_terminate_mps: Optional[float] = None,
        termination_grace_steps: int = 16,
        attitude_violation_steps: int = 3,
        overspeed_violation_steps: int = 3,
        hard_attitude_terminate_rad: Optional[float] = None,
        hard_attitude_violation_steps: Optional[int] = None,
        base_height_terminate: Optional[float] = None,
        base_height_grace: Optional[float] = None,
        command_ramp_steps: Optional[int] = None,
        startup_max_cmd_mps: Optional[float] = None,
        startup_torque_scale_steps: Optional[int] = None,
        startup_torque_scale: Optional[float] = None,
        use_two_stage_curriculum: bool = False,
        use_six_stage_curriculum: bool = True,
        curriculum_stage_1_min: Optional[float] = None,
        curriculum_stage_1_max: Optional[float] = None,
        curriculum_stage_2_min: Optional[float] = None,
        curriculum_stage_2_max: Optional[float] = None,
        curriculum_stage_3_min: Optional[float] = None,
        curriculum_stage_3_max: Optional[float] = None,
        curriculum_stage_4_min: Optional[float] = None,
        curriculum_stage_4_max: Optional[float] = None,
        curriculum_stage_5_min: Optional[float] = None,
        curriculum_stage_5_max: Optional[float] = None,
        curriculum_stage_6_min: Optional[float] = None,
        curriculum_stage_6_max: Optional[float] = None,
        curriculum_early_stair_start_x_min: Optional[float] = None,
        curriculum_early_stair_start_x_max: Optional[float] = None,
        curriculum_early_stair_height_min: Optional[float] = None,
        curriculum_early_stair_height_max: Optional[float] = None,
        curriculum_late_stair_start_x_min: Optional[float] = None,
        curriculum_late_stair_start_x_max: Optional[float] = None,
        curriculum_late_stair_height_min: Optional[float] = None,
        curriculum_late_stair_height_max: Optional[float] = None,
        curriculum_success_threshold: Optional[float] = None,
        curriculum_window_episodes: Optional[int] = None,
        curriculum_progression_criterion: str = "course_complete",
        speed_curriculum_enabled: Optional[bool] = None,
        speed_curriculum_min: Optional[float] = None,
        speed_curriculum_max: Optional[float] = None,
        speed_curriculum_warmup_episodes: Optional[int] = None,
        speed_curriculum_full_episodes: Optional[int] = None,
        random_command_enabled: Optional[bool] = None,
        random_command_min: Optional[float] = None,
        random_command_max: Optional[float] = None,
        randomize_stairs: Optional[bool] = None,
        stair_start_x_min: Optional[float] = None,
        stair_start_x_max: Optional[float] = None,
        stair_step_height_min: Optional[float] = None,
        stair_step_height_max: Optional[float] = None,
        stair_step_width_min: Optional[float] = None,
        stair_step_width_max: Optional[float] = None,
        stair_num_steps_up_min: Optional[int] = None,
        stair_num_steps_up_max: Optional[int] = None,
        stair_num_steps_down_min: Optional[int] = None,
        stair_num_steps_down_max: Optional[int] = None,
        stair_equal_up_down_steps: Optional[bool] = None,
        multi_obstacles_enabled: bool = True,
        extra_obstacle_sets: int = 2,
        obstacle_gap_min: float = 0.8,
        obstacle_gap_max: float = 1.8,
        success_reset_enabled: bool = True,
        success_reset_hold_steps: int = 20,
        success_reset_min_episode_step: int = 120,
        success_reset_x_margin_m: float = 0.10,
        post_reset_stand_steps: int = 15,
        reset_recovery_base_z_min: float = 0.20,
        fixed_course: bool = True,
        stagnation_window_steps: int = 300,
        stagnation_threshold_m: float = 0.05, 
        yaw_hold_penalty: float = 0.25,
        learning_starts_steps: int = 100_000,
        course_complete_terminal_bonus: float = 10000.0,
        attitude_hard_terminal_penalty: float = 1000.0,
        stagnation_terminal_penalty: float = 1000.0,
    ) -> None:
        super().__init__()
        _ensure_casadi_binary_path()

        required_args = {
            'headless': headless,
            'frame_skip': frame_skip,
            'episode_length': episode_length,
            'command_forward': command_forward,
            'command_lateral': command_lateral,
            'command_yaw': command_yaw,
            'command_pitch': command_pitch,
            'stair_start_x': stair_start_x,
            'residual_warmup_steps': residual_warmup_steps,
            'residual_ramp_steps': residual_ramp_steps,
            'overspeed_terminate_mps': overspeed_terminate_mps,
            'hard_attitude_terminate_rad': hard_attitude_terminate_rad,
            'hard_attitude_violation_steps': hard_attitude_violation_steps,
            'base_height_terminate': base_height_terminate,
            'base_height_grace': base_height_grace,
            'command_ramp_steps': command_ramp_steps,
            'startup_max_cmd_mps': startup_max_cmd_mps,
            'startup_torque_scale_steps': startup_torque_scale_steps,
            'startup_torque_scale': startup_torque_scale,
            'speed_curriculum_enabled': speed_curriculum_enabled,
            'speed_curriculum_min': speed_curriculum_min,
            'speed_curriculum_max': speed_curriculum_max,
            'speed_curriculum_warmup_episodes': speed_curriculum_warmup_episodes,
            'speed_curriculum_full_episodes': speed_curriculum_full_episodes,
            'randomize_stairs': randomize_stairs,
            'stair_start_x_min': stair_start_x_min,
            'stair_start_x_max': stair_start_x_max,
            'stair_step_height_min': stair_step_height_min,
            'stair_step_height_max': stair_step_height_max,
            'stair_step_width_min': stair_step_width_min,
            'stair_step_width_max': stair_step_width_max,
            'stair_num_steps_up_min': stair_num_steps_up_min,
            'stair_num_steps_up_max': stair_num_steps_up_max,
            'stair_num_steps_down_min': stair_num_steps_down_min,
            'stair_num_steps_down_max': stair_num_steps_down_max,
            'stair_equal_up_down_steps': stair_equal_up_down_steps,
        }
        missing_args = [name for name, value in required_args.items() if value is None]
        if missing_args:
            raise ValueError(
                "IsaacStepHeightResidualEnv requires explicit config for training-controlled settings: "
                + ", ".join(missing_args)
            )

        self.frame_skip = int(frame_skip)
        self.episode_length = int(episode_length)
        self.command_forward = float(command_forward)
        self.command_lateral = float(command_lateral)
        self.command_yaw = float(command_yaw)
        self.command_pitch = float(command_pitch)
        self.stair_start_x = float(stair_start_x)
        self.residual_warmup_steps = int(residual_warmup_steps)
        self.residual_ramp_steps = int(residual_ramp_steps)
        self.overspeed_terminate_mps = float(overspeed_terminate_mps)
        self.termination_grace_steps = int(max(0, termination_grace_steps))
        self.attitude_violation_steps = int(max(1, attitude_violation_steps))
        self.overspeed_violation_steps = int(max(1, overspeed_violation_steps))
        self.hard_attitude_terminate_rad = float(max(0.8, hard_attitude_terminate_rad))
        self.hard_attitude_violation_steps = int(max(1, hard_attitude_violation_steps))
        self.base_height_terminate = float(base_height_terminate)
        self.base_height_grace = float(base_height_grace)
        self.command_ramp_steps = int(max(0, command_ramp_steps))
        self.startup_max_cmd_mps = float(startup_max_cmd_mps)
        self.startup_torque_scale_steps = int(max(0, startup_torque_scale_steps))
        self.startup_torque_scale = float(np.clip(startup_torque_scale, 0.0, 1.0))
        self.use_two_stage_curriculum = bool(use_two_stage_curriculum)
        self.use_six_stage_curriculum = bool(use_six_stage_curriculum)
        
        # 2-stage curriculum (forward speed only, yaw always 0)
        self.curriculum_stage_1_min = float(curriculum_stage_1_min if curriculum_stage_1_min is not None else 0.4)
        self.curriculum_stage_1_max = float(curriculum_stage_1_max if curriculum_stage_1_max is not None else 0.7)
        self.curriculum_stage_2_min = float(curriculum_stage_2_min if curriculum_stage_2_min is not None else 0.7)
        self.curriculum_stage_2_max = float(curriculum_stage_2_max if curriculum_stage_2_max is not None else 1.0)
        self.curriculum_stage_3_min = float(curriculum_stage_3_min if curriculum_stage_3_min is not None else self.curriculum_stage_2_min)
        self.curriculum_stage_3_max = float(curriculum_stage_3_max if curriculum_stage_3_max is not None else self.curriculum_stage_2_max)
        self.curriculum_stage_4_min = float(curriculum_stage_4_min if curriculum_stage_4_min is not None else self.curriculum_stage_1_min)
        self.curriculum_stage_4_max = float(curriculum_stage_4_max if curriculum_stage_4_max is not None else self.curriculum_stage_1_max)
        self.curriculum_stage_5_min = float(curriculum_stage_5_min if curriculum_stage_5_min is not None else self.curriculum_stage_2_min)
        self.curriculum_stage_5_max = float(curriculum_stage_5_max if curriculum_stage_5_max is not None else self.curriculum_stage_2_max)
        self.curriculum_stage_6_min = float(curriculum_stage_6_min if curriculum_stage_6_min is not None else self.curriculum_stage_3_min)
        self.curriculum_stage_6_max = float(curriculum_stage_6_max if curriculum_stage_6_max is not None else self.curriculum_stage_3_max)
        self.curriculum_early_stair_start_x_min = float(curriculum_early_stair_start_x_min if curriculum_early_stair_start_x_min is not None else 1.2)
        self.curriculum_early_stair_start_x_max = float(curriculum_early_stair_start_x_max if curriculum_early_stair_start_x_max is not None else 1.5)
        self.curriculum_early_stair_height_min = float(curriculum_early_stair_height_min if curriculum_early_stair_height_min is not None else 0.08)
        self.curriculum_early_stair_height_max = float(curriculum_early_stair_height_max if curriculum_early_stair_height_max is not None else 0.12)
        self.curriculum_late_stair_start_x_min = float(curriculum_late_stair_start_x_min if curriculum_late_stair_start_x_min is not None else 1.5)
        self.curriculum_late_stair_start_x_max = float(curriculum_late_stair_start_x_max if curriculum_late_stair_start_x_max is not None else 2.0)
        self.curriculum_late_stair_height_min = float(curriculum_late_stair_height_min if curriculum_late_stair_height_min is not None else 0.12)
        self.curriculum_late_stair_height_max = float(curriculum_late_stair_height_max if curriculum_late_stair_height_max is not None else 0.16)
        self.curriculum_success_threshold = float(curriculum_success_threshold if curriculum_success_threshold is not None else 0.60)
        self.curriculum_window_episodes = int(curriculum_window_episodes if curriculum_window_episodes is not None else 10)
        criterion = str(curriculum_progression_criterion).strip().lower()
        if criterion not in ("course_complete", "course_success"):
            criterion = "course_complete"
        self.curriculum_progression_criterion = criterion
        self._curriculum_stage = 1  # Start at stage 1
        self._curriculum_stage_success = deque(maxlen=self.curriculum_window_episodes)
        self._six_stage_specs = [
            {
                'speed_min': self.curriculum_stage_1_min,
                'speed_max': self.curriculum_stage_1_max,
                'stair_start_x_min': self.curriculum_early_stair_start_x_min,
                'stair_start_x_max': self.curriculum_early_stair_start_x_max,
                'stair_height_min': self.curriculum_early_stair_height_min,
                'stair_height_max': self.curriculum_early_stair_height_max,
            },
            {
                'speed_min': self.curriculum_stage_2_min,
                'speed_max': self.curriculum_stage_2_max,
                'stair_start_x_min': self.curriculum_early_stair_start_x_min,
                'stair_start_x_max': self.curriculum_early_stair_start_x_max,
                'stair_height_min': self.curriculum_early_stair_height_min,
                'stair_height_max': self.curriculum_early_stair_height_max,
            },
            {
                'speed_min': self.curriculum_stage_3_min,
                'speed_max': self.curriculum_stage_3_max,
                'stair_start_x_min': self.curriculum_early_stair_start_x_min,
                'stair_start_x_max': self.curriculum_early_stair_start_x_max,
                'stair_height_min': self.curriculum_early_stair_height_min,
                'stair_height_max': self.curriculum_early_stair_height_max,
            },
            {
                'speed_min': self.curriculum_stage_4_min,
                'speed_max': self.curriculum_stage_4_max,
                'stair_start_x_min': self.curriculum_late_stair_start_x_min,
                'stair_start_x_max': self.curriculum_late_stair_start_x_max,
                'stair_height_min': self.curriculum_late_stair_height_min,
                'stair_height_max': self.curriculum_late_stair_height_max,
            },
            {
                'speed_min': self.curriculum_stage_5_min,
                'speed_max': self.curriculum_stage_5_max,
                'stair_start_x_min': self.curriculum_late_stair_start_x_min,
                'stair_start_x_max': self.curriculum_late_stair_start_x_max,
                'stair_height_min': self.curriculum_late_stair_height_min,
                'stair_height_max': self.curriculum_late_stair_height_max,
            },
            {
                'speed_min': self.curriculum_stage_6_min,
                'speed_max': self.curriculum_stage_6_max,
                'stair_start_x_min': self.curriculum_late_stair_start_x_min,
                'stair_start_x_max': self.curriculum_late_stair_start_x_max,
                'stair_height_min': self.curriculum_late_stair_height_min,
                'stair_height_max': self.curriculum_late_stair_height_max,
            },
        ]
        
        self.speed_curriculum_enabled = bool(speed_curriculum_enabled)
        self.speed_curriculum_min = float(speed_curriculum_min)
        self.speed_curriculum_max = float(max(0.30, speed_curriculum_max))
        self.speed_curriculum_warmup_episodes = int(max(0, speed_curriculum_warmup_episodes))
        self.speed_curriculum_full_episodes = int(max(self.speed_curriculum_warmup_episodes + 1, speed_curriculum_full_episodes))
        self.random_command_enabled = bool(random_command_enabled)
        self.random_command_min = float(min(random_command_min, random_command_max))
        self.random_command_max = float(max(random_command_min, random_command_max))
        self.randomize_stairs = bool(randomize_stairs)
        self.stair_start_x_min = float(min(stair_start_x_min, stair_start_x_max))
        self.stair_start_x_max = float(max(stair_start_x_min, stair_start_x_max))
        self.stair_step_height_min = float(min(stair_step_height_min, stair_step_height_max))
        self.stair_step_height_max = float(max(stair_step_height_min, stair_step_height_max))
        self.stair_step_width_min = float(min(stair_step_width_min, stair_step_width_max))
        self.stair_step_width_max = float(max(stair_step_width_min, stair_step_width_max))
        self.stair_num_steps_up_min = int(min(stair_num_steps_up_min, stair_num_steps_up_max))
        self.stair_num_steps_up_max = int(max(stair_num_steps_up_min, stair_num_steps_up_max))
        self.stair_num_steps_down_min = int(min(stair_num_steps_down_min, stair_num_steps_down_max))
        self.stair_num_steps_down_max = int(max(stair_num_steps_down_min, stair_num_steps_down_max))
        self.stair_equal_up_down_steps = bool(stair_equal_up_down_steps)
        self.multi_obstacles_enabled = bool(multi_obstacles_enabled)
        self.extra_obstacle_sets = int(max(0, extra_obstacle_sets))
        self.obstacle_gap_min = float(min(obstacle_gap_min, obstacle_gap_max))
        self.obstacle_gap_max = float(max(obstacle_gap_min, obstacle_gap_max))
        self.success_reset_enabled = bool(success_reset_enabled)
        self.success_reset_hold_steps = int(max(1, success_reset_hold_steps))
        self.success_reset_min_episode_step = int(max(0, success_reset_min_episode_step))
        self.success_reset_x_margin_m = float(max(0.0, success_reset_x_margin_m))
        self.post_reset_stand_steps = int(max(0, post_reset_stand_steps))
        self.reset_recovery_base_z_min = float(max(0.05, reset_recovery_base_z_min))
        self.fixed_course = bool(fixed_course)
        self.stagnation_window_steps = int(max(50, stagnation_window_steps))
        # Increased from 200 to 280: gives more recovery time after stumbles at higher speeds.
        # At 0.517 m/s where backward slides occur, robot needs ~2.8s (280 steps) to recover.
        if self.stagnation_window_steps == 200:  # Only update the default; respect explicit user overrides
            self.stagnation_window_steps = 280
        self.stagnation_threshold_m = float(max(0.01, stagnation_threshold_m))
        self.yaw_hold_penalty = float(max(0.0, yaw_hold_penalty))
        self.learning_starts_steps = int(max(0, learning_starts_steps))
        self.course_complete_terminal_bonus = float(max(0.0, course_complete_terminal_bonus))
        self.attitude_hard_terminal_penalty = float(max(0.0, attitude_hard_terminal_penalty))
        self.stagnation_terminal_penalty = float(max(0.0, stagnation_terminal_penalty))

        # Curriculum gating: only increase difficulty when recent success is stable.
        self.curriculum_gate_enabled = True
        self.curriculum_gate_window = 10
        self.curriculum_gate_success_threshold = 0.60
        self.curriculum_progress_step = 1.0 / max(1.0, float(self.speed_curriculum_full_episodes - self.speed_curriculum_warmup_episodes))
        self.curriculum_progress_decay = 0.5 * self.curriculum_progress_step
        self.curriculum_downgrade_threshold = 0.35  # If success falls below this, allow curriculum to backtrack
        self._curriculum_progress = 0.0
        self._recent_success = deque(maxlen=self.curriculum_gate_window)

        self._episode_step = 0
        self._cumulative_env_steps = 0  # Track total steps across all episodes for learning_starts gating
        self._render_enabled = not bool(headless)

        from isaacsim import SimulationApp

        self._simulation_app = SimulationApp({"headless": bool(headless)})

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        pympc_root = os.path.abspath(os.path.join(project_root, "..", "pympc"))
        if pympc_root not in sys.path:
            sys.path.insert(0, pympc_root)

        from src.dog.dogWorld import DogWorld
        from src.dog.dog_runner import DogRunner, SimulationState
        from quadruped_pympc import config as cfg

        self._SimulationState = SimulationState
        self._cfg = cfg

        # Enable RL residual assist in this env
        self._cfg.simulation_params['rl_step_height_enabled'] = True
        # Save/override reflex and retry policies for training-phase control.
        # Policy:
        # - both pre-learning and learning: retry OFF, reflex OFF,
        #   next-step enhancement OFF so RL residual fully owns step-height
        #   adaptation during training.
        self._orig_obstacle_retry_enabled = bool(self._cfg.simulation_params.get('obstacle_retry_enabled', False))
        self._orig_reflex_trigger_mode = self._cfg.simulation_params.get('reflex_trigger_mode', False)
        self._orig_reflex_next_steps_height_enhancement = bool(
            self._cfg.simulation_params.get('reflex_next_steps_height_enhancement', False)
        )
        self._cfg.simulation_params['obstacle_retry_enabled'] = False
        self._cfg.simulation_params['reflex_trigger_mode'] = False
        self._cfg.simulation_params['reflex_next_steps_height_enhancement'] = False

        world_cfg = DogWorld.Params()
        world_cfg.add_stairs = bool(add_stairs)
        if self.fixed_course and bool(add_stairs):
            # Fixed deterministic obstacle course
            world_cfg.stair_start_x = 1.5
            world_cfg.stair_step_height = 0.08
            world_cfg.stair_step_width = 0.30
            world_cfg.stair_num_steps_up = 2
            world_cfg.stair_num_steps_down = 2
        elif self.randomize_stairs and bool(add_stairs):
            sampled = self._sample_stair_geometry()
            world_cfg.stair_start_x = sampled['stair_start_x']
            world_cfg.stair_step_height = sampled['stair_step_height']
            world_cfg.stair_step_width = sampled['stair_step_width']
            world_cfg.stair_num_steps_up = sampled['stair_num_steps_up']
            world_cfg.stair_num_steps_down = sampled['stair_num_steps_down']
        else:
            world_cfg.stair_start_x = float(stair_start_x)
            if self.stair_equal_up_down_steps and bool(add_stairs):
                world_cfg.stair_num_steps_down = int(world_cfg.stair_num_steps_up)
        self._stair_step_height = float(world_cfg.stair_step_height)
        self._stair_step_width = float(world_cfg.stair_step_width)
        self._stair_num_steps_up = int(world_cfg.stair_num_steps_up)
        self._stair_num_steps_down = int(world_cfg.stair_num_steps_down)
        self.stair_start_x = float(world_cfg.stair_start_x)
        self._stair_top_height = self._stair_step_height * self._stair_num_steps_up
        self._stair_up_end_x = self.stair_start_x + self._stair_num_steps_up * self._stair_step_width
        self._stair_down_end_x = self._stair_up_end_x + self._stair_num_steps_down * self._stair_step_width
        self._course_end_x = float(self._stair_down_end_x)
        self._descend_visual_drop_m = max(0.06, 0.5 * self._stair_step_height * self._stair_num_steps_down)
        self._descend_flat_z_margin_m = max(0.08, self._stair_step_height + 0.04)
        self._descend_down_end_x_margin_m = max(0.05, 0.75 * self._stair_step_width)
        self._world = DogWorld(world_cfg)
        self._runner = DogRunner(self._world)
        self._runner.state = self._SimulationState.ACTIVE
        # Configure runtime safety knobs on the runner to reduce startup spikes.
        # NOTE: we intentionally do NOT set 'startup_use_standing' – that attribute
        # replaces MPC torques with standing-controller torques which blocks forward
        # motion.  Instead we rely on gentle torque scaling for the first few steps.
        try:
            setattr(self._runner, 'startup_torque_scale_steps', int(self.startup_torque_scale_steps))
            setattr(self._runner, 'startup_torque_scale', float(self.startup_torque_scale))
        except Exception:
            pass

        self._episode_max_x = 0.0
        self._episode_max_z = 0.0
        self._episode_start_z = 0.0
        self._episode_climbed_peak = False
        self._episode_descended_to_flat = False
        self._success_post_steps = 0
        self._episode_count = 0
        self._episode_command_forward = float(self.command_forward)
        self._last_residual_gate = 0.0
        self._tilt_violation_count = 0
        self._hard_tilt_violation_count = 0
        self._overspeed_violation_count = 0
        # Completion time tracking
        self._course_entry_step = -1  # step when robot crosses stair_start_x
        self._course_complete_step = -1  # step when robot passes course_end_x after descending
        # Stagnation tracking
        self._stagnation_x_history = deque(maxlen=self.stagnation_window_steps)
        # Fixed course length (constant)
        self._course_length = 0.0
        # Yaw maintenance tracking
        self._episode_start_yaw = None
        self._episode_max_abs_yaw_error = 0.0
        # Deviation termination: if the robot advances too far without actually
        # stepping onto/climbing the stair sequence, end the episode.
        self.deviation_no_stair_distance_m = 3.0
        # Also terminate when the robot veers too far sideways from the stair
        # centerline before climbing (typical bypass behavior).
        step_depth = float(getattr(self._world.params, 'stair_step_depth', 3.0))
        self.deviation_lateral_no_stair_m = max(0.6, 0.30 * step_depth)
        self.deviation_lateral_activation_x_margin_m = 0.25

        self._max_residual_m = float(self._cfg.simulation_params.get('rl_step_height_residual_max_abs', 0.08))
        self._nominal_swing_time_s = float(self._cfg.simulation_params.get('adaptive_swing_time', 0.20))
        self._min_swing_time_s = float(self._cfg.simulation_params.get('adaptive_swing_time_min', 0.16))
        self._max_swing_time_s = float(self._cfg.simulation_params.get('adaptive_swing_time_max', 0.22))
        self._prev_residual_m = 0.0
        self._prev_swing_time_s = float(self._nominal_swing_time_s)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        obs = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs.shape,
            dtype=np.float32,
        )

    def _sample_obstacle_sets(self) -> list[dict]:
        sets = []
        if (not self.multi_obstacles_enabled) or self.extra_obstacle_sets <= 0:
            return sets

        cursor_x = float(self._stair_down_end_x)
        for _ in range(self.extra_obstacle_sets):
            gap = float(np.random.uniform(self.obstacle_gap_min, self.obstacle_gap_max))
            stair_num_steps_up = int(np.random.randint(self.stair_num_steps_up_min, self.stair_num_steps_up_max + 1))
            if self.stair_equal_up_down_steps:
                stair_num_steps_down = stair_num_steps_up
            else:
                stair_num_steps_down = int(np.random.randint(self.stair_num_steps_down_min, self.stair_num_steps_down_max + 1))
            stair_step_width = float(np.random.uniform(self.stair_step_width_min, self.stair_step_width_max))
            stair_step_height = float(np.random.uniform(self.stair_step_height_min, self.stair_step_height_max))
            stair_start_x = float(cursor_x + gap)

            up_end_x = float(stair_start_x + stair_num_steps_up * stair_step_width)
            down_end_x = float(up_end_x + stair_num_steps_down * stair_step_width)
            cursor_x = down_end_x

            sets.append(
                {
                    'stair_start_x': stair_start_x,
                    'stair_step_height': stair_step_height,
                    'stair_step_width': stair_step_width,
                    'stair_num_steps_up': stair_num_steps_up,
                    'stair_num_steps_down': stair_num_steps_down,
                    'stair_up_end_x': up_end_x,
                    'stair_down_end_x': down_end_x,
                }
            )
        return sets

    def _build_fixed_obstacle_sets(self) -> list[dict]:
        """Build the two extra deterministic obstacle sets for the fixed course.

        Fixed course layout (set 1 is built by DogWorld as main stairs):
          Set 1: 2 up + 2 down, 0.12 m height, 0.30 m width, starts at 1.5
          Set 2: 3 up + 3 down, 0.10 m height, 0.30 m width
          Set 3: 4 up + 4 down, 0.08 m height, 0.30 m width
          Gap between sets: 1.0 m
        """
        STEP_WIDTH = 0.30
        GAP = 1.0
        sets = []
        cursor_x = float(self._stair_down_end_x)

        # Set 2: 3 up + 3 down, 10 cm
        start2 = cursor_x + GAP
        up_end2 = start2 + 3 * STEP_WIDTH
        down_end2 = up_end2 + 3 * STEP_WIDTH
        sets.append({
            'stair_start_x': start2,
            'stair_step_height': 0.10,
            'stair_step_width': STEP_WIDTH,
            'stair_num_steps_up': 3,
            'stair_num_steps_down': 3,
            'stair_up_end_x': up_end2,
            'stair_down_end_x': down_end2,
        })

        # Set 3: 4 up + 4 down, 8 cm
        start3 = down_end2 + GAP
        up_end3 = start3 + 4 * STEP_WIDTH
        down_end3 = up_end3 + 4 * STEP_WIDTH
        sets.append({
            'stair_start_x': start3,
            'stair_step_height': 0.12,
            'stair_step_width': STEP_WIDTH,
            'stair_num_steps_up': 4,
            'stair_num_steps_down': 4,
            'stair_up_end_x': up_end3,
            'stair_down_end_x': down_end3,
        })

        return sets

    def _apply_obstacle_sets(self, obstacle_sets: list[dict]) -> None:
        from pxr import PhysxSchema
        from isaacsim.core.api.materials import PhysicsMaterial
        from isaacsim.core.api.objects import FixedCuboid

        stage = self._world.stage
        if stage.GetPrimAtPath("/World/extra_obstacles"):
            stage.RemovePrim("/World/extra_obstacles")

        if len(obstacle_sets) == 0:
            self._course_end_x = float(self._stair_down_end_x)
            return

        step_depth = float(getattr(self._world.params, 'stair_step_depth', 3.0))
        stair_material_path = "/World/Physics_Materials/StairMaterial"
        stair_material = PhysicsMaterial(
            stair_material_path,
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.0,
        )
        stair_material_prim = stage.GetPrimAtPath(stair_material_path)
        PhysxSchema.PhysxMaterialAPI.Apply(stair_material_prim).CreateFrictionCombineModeAttr().Set("max")

        for set_idx, obstacle in enumerate(obstacle_sets):
            stair_start_x = float(obstacle['stair_start_x'])
            stair_step_height = float(obstacle['stair_step_height'])
            stair_step_width = float(obstacle['stair_step_width'])
            stair_num_steps_up = int(obstacle['stair_num_steps_up'])
            stair_num_steps_down = int(obstacle['stair_num_steps_down'])

            for i in range(stair_num_steps_up):
                x_pos = float(stair_start_x + i * stair_step_width)
                z_pos = float(stair_step_height / 2 + i * stair_step_height)
                step = FixedCuboid(
                    prim_path=f"/World/extra_obstacles/set_{set_idx}/up_step_{i}",
                    position=np.array([x_pos, 0.0, z_pos], dtype=np.float64),
                    size=1.0,
                    scale=np.array([stair_step_width, step_depth, stair_step_height], dtype=np.float64),
                    color=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                )
                step.apply_physics_material(stair_material)

            down_start_x = float(stair_start_x + stair_num_steps_up * stair_step_width)
            initial_height = float(stair_step_height * stair_num_steps_up)
            for i in range(stair_num_steps_down):
                x_pos = float(down_start_x + i * stair_step_width)
                z_pos = float(initial_height - (stair_step_height / 2 + i * stair_step_height))
                step = FixedCuboid(
                    prim_path=f"/World/extra_obstacles/set_{set_idx}/down_step_{i}",
                    position=np.array([x_pos, 0.0, z_pos], dtype=np.float64),
                    size=1.0,
                    scale=np.array([stair_step_width, step_depth, stair_step_height], dtype=np.float64),
                    color=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                )
                step.apply_physics_material(stair_material)

        self._course_end_x = float(obstacle_sets[-1]['stair_down_end_x'])
        self._course_length = float(self._course_end_x - self.stair_start_x)

    def _sample_stair_geometry(self) -> dict:
        if self.use_six_stage_curriculum:
            idx = int(np.clip(self._curriculum_stage - 1, 0, len(self._six_stage_specs) - 1))
            spec = self._six_stage_specs[idx]
            stage_stair_start_x_min = float(min(spec['stair_start_x_min'], spec['stair_start_x_max']))
            stage_stair_start_x_max = float(max(spec['stair_start_x_min'], spec['stair_start_x_max']))
            stage_stair_height_min = float(min(spec['stair_height_min'], spec['stair_height_max']))
            stage_stair_height_max = float(max(spec['stair_height_min'], spec['stair_height_max']))
        else:
            stage_stair_start_x_min = self.stair_start_x_min
            stage_stair_start_x_max = self.stair_start_x_max
            stage_stair_height_min = self.stair_step_height_min
            stage_stair_height_max = self.stair_step_height_max

        stair_num_steps_up = int(np.random.randint(self.stair_num_steps_up_min, self.stair_num_steps_up_max + 1))
        if self.stair_equal_up_down_steps:
            stair_num_steps_down = stair_num_steps_up
        else:
            stair_num_steps_down = int(np.random.randint(self.stair_num_steps_down_min, self.stair_num_steps_down_max + 1))

        return {
            'stair_start_x': float(np.random.uniform(stage_stair_start_x_min, stage_stair_start_x_max)),
            'stair_step_height': float(np.random.uniform(stage_stair_height_min, stage_stair_height_max)),
            'stair_step_width': float(np.random.uniform(self.stair_step_width_min, self.stair_step_width_max)),
            'stair_num_steps_up': int(stair_num_steps_up),
            'stair_num_steps_down': int(stair_num_steps_down),
        }

    def _apply_stair_geometry(self, geometry: dict) -> None:
        from pxr import PhysxSchema
        from isaacsim.core.api.materials import PhysicsMaterial
        from isaacsim.core.api.objects import FixedCuboid

        stage = self._world.stage
        step_depth = float(getattr(self._world.params, 'stair_step_depth', 3.0))

        # Remove previously created stair prims so geometry can be refreshed.
        for i in range(64):
            up_path = f"/World/stairs_up_step_{i}"
            down_path = f"/World/stairs_down_step_{i}"
            if stage.GetPrimAtPath(up_path):
                stage.RemovePrim(up_path)
            if stage.GetPrimAtPath(down_path):
                stage.RemovePrim(down_path)

        stair_start_x = float(geometry['stair_start_x'])
        stair_step_height = float(geometry['stair_step_height'])
        stair_step_width = float(geometry['stair_step_width'])
        stair_num_steps_up = int(geometry['stair_num_steps_up'])
        stair_num_steps_down = int(geometry['stair_num_steps_down'])

        self._world.params.stair_start_x = stair_start_x
        self._world.params.stair_step_height = stair_step_height
        self._world.params.stair_step_width = stair_step_width
        self._world.params.stair_num_steps_up = stair_num_steps_up
        self._world.params.stair_num_steps_down = stair_num_steps_down

        stair_material_path = "/World/Physics_Materials/StairMaterial"
        stair_material = PhysicsMaterial(
            stair_material_path,
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.0,
        )
        stair_material_prim = stage.GetPrimAtPath(stair_material_path)
        PhysxSchema.PhysxMaterialAPI.Apply(stair_material_prim).CreateFrictionCombineModeAttr().Set("max")

        for i in range(stair_num_steps_up):
            x_pos = float(stair_start_x + i * stair_step_width)
            z_pos = float(stair_step_height / 2 + i * stair_step_height)
            step = FixedCuboid(
                prim_path=f"/World/stairs_up_step_{i}",
                position=np.array([x_pos, 0.0, z_pos], dtype=np.float64),
                size=1.0,
                scale=np.array([stair_step_width, step_depth, stair_step_height], dtype=np.float64),
                color=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            )
            step.apply_physics_material(stair_material)

        down_start_x = float(stair_start_x + stair_num_steps_up * stair_step_width)
        initial_height = float(stair_step_height * stair_num_steps_up)
        for i in range(stair_num_steps_down):
            x_pos = float(down_start_x + i * stair_step_width)
            z_pos = float(initial_height - (stair_step_height / 2 + i * stair_step_height))
            step = FixedCuboid(
                prim_path=f"/World/stairs_down_step_{i}",
                position=np.array([x_pos, 0.0, z_pos], dtype=np.float64),
                size=1.0,
                scale=np.array([stair_step_width, step_depth, stair_step_height], dtype=np.float64),
                color=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            )
            step.apply_physics_material(stair_material)

        self._stair_step_height = stair_step_height
        self._stair_step_width = stair_step_width
        self._stair_num_steps_up = stair_num_steps_up
        self._stair_num_steps_down = stair_num_steps_down
        self.stair_start_x = stair_start_x
        self._stair_top_height = self._stair_step_height * self._stair_num_steps_up
        self._stair_up_end_x = self.stair_start_x + self._stair_num_steps_up * self._stair_step_width
        self._stair_down_end_x = self._stair_up_end_x + self._stair_num_steps_down * self._stair_step_width
        self._course_end_x = float(self._stair_down_end_x)
        self._descend_visual_drop_m = max(0.06, 0.5 * self._stair_step_height * self._stair_num_steps_down)
        self._descend_flat_z_margin_m = max(0.08, self._stair_step_height + 0.04)
        self._descend_down_end_x_margin_m = max(0.05, 0.75 * self._stair_step_width)

        self._apply_obstacle_sets(self._sample_obstacle_sets())

    def _set_command(self) -> None:
        # Apply a startup ramp to commands to avoid aggressive initial transients
        ramp = 1.0
        if self.command_ramp_steps > 0:
            ramp = float(np.clip(self._episode_step / float(self.command_ramp_steps), 0.0, 1.0))

        # Blend from a startup-capped command to the full ramped command so there is
        # no hard jump when the startup window ends.
        desired_vx = self._episode_command_forward * ramp
        startup_blend = 1.0
        if self.command_ramp_steps > 0:
            startup_blend = float(np.clip(self._episode_step / float(self.command_ramp_steps), 0.0, 1.0))
        capped_vx = float(np.clip(desired_vx, -self.startup_max_cmd_mps, self.startup_max_cmd_mps))
        if self.command_ramp_steps > 0:
            effective_vx = float((1.0 - startup_blend) * capped_vx + startup_blend * desired_vx)
        else:
            effective_vx = float(desired_vx)

        self._runner.ref_base_lin_vel_H[0] = effective_vx
        self._runner.ref_base_lin_vel_H[1] = self.command_lateral * ramp
        self._runner.ref_base_lin_vel_H[2] = 0.0
        self._runner.ref_base_ang_yaw_dot = self.command_yaw * ramp
        self._runner.pitch_delta = self.command_pitch * ramp
        if hasattr(self._runner.mpc_controller, 'set_reference_pitch_offset'):
            self._runner.mpc_controller.set_reference_pitch_offset(self.command_pitch * ramp)

    def _sample_episode_command_forward(self) -> float:
        # Preserve legacy random-command mode for future experiments.
        if self.random_command_enabled:
            return float(np.random.uniform(self.random_command_min, self.random_command_max))

        if self.use_six_stage_curriculum:
            idx = int(np.clip(self._curriculum_stage - 1, 0, len(self._six_stage_specs) - 1))
            spec = self._six_stage_specs[idx]
            return float(np.random.uniform(float(spec['speed_min']), float(spec['speed_max'])))

        if not self.use_two_stage_curriculum:
            return float(self.command_forward)

        # 2-stage curriculum: sample forward command from current stage
        # Stage 1: [0.4, 0.7]
        # Stage 2: [0.7, 1.0]
        # Advance to stage 2 when success >= 60% over 10 episodes
        if self._curriculum_stage == 1:
            stage_min = self.curriculum_stage_1_min
            stage_max = self.curriculum_stage_1_max
        else:
            stage_min = self.curriculum_stage_2_min
            stage_max = self.curriculum_stage_2_max
        
        return float(np.random.uniform(stage_min, stage_max))

    def _update_curriculum_progress(self, episode_success: bool) -> None:
        if self.use_six_stage_curriculum:
            if self._curriculum_stage >= len(self._six_stage_specs):
                return

            self._curriculum_stage_success.append(1 if episode_success else 0)
            if len(self._curriculum_stage_success) < self.curriculum_window_episodes:
                return

            rolling_success_rate = float(np.mean(self._curriculum_stage_success))
            if rolling_success_rate >= self.curriculum_success_threshold:
                self._curriculum_stage += 1
                self._curriculum_stage_success.clear()
                print(f"[Curriculum] Advanced to stage {self._curriculum_stage} (success rate: {rolling_success_rate:.2%})")
            return

        if not self.use_two_stage_curriculum:
            return

        # 2-stage curriculum: track rolling success and advance stage when threshold met
        # Stage 1: forward command in [0.4, 0.7] m/s
        # Stage 2: forward command in [0.7, 1.0] m/s
        # Advance to stage 2 when success rate >= 60% over last 10 episodes
        if self._curriculum_stage >= 2:
            # Already at final stage, no advancement needed
            return
        
        # Record success for this episode
        self._curriculum_stage_success.append(1 if episode_success else 0)
        
        # Only check for advancement when deque is full (has curriculum_window_episodes entries)
        if len(self._curriculum_stage_success) < self.curriculum_window_episodes:
            return
        
        # Compute rolling success rate
        rolling_success_rate = float(np.mean(self._curriculum_stage_success))
        
        # Check if we should advance to next stage
        if rolling_success_rate >= self.curriculum_success_threshold:
            self._curriculum_stage += 1
            self._curriculum_stage_success.clear()  # Reset tracking for new stage
            print(f"[Curriculum] Advanced to stage {self._curriculum_stage} (success rate: {rolling_success_rate:.2%})")

    def _advance_once(self) -> None:
        self._runner.pre_physics()
        self._runner.step_physics()
        self._runner.stage.world.step(render=self._render_enabled)
        self._runner.post_physics()

    def _post_reset_recovery(self) -> None:
        # Keep the robot in INIT/standing mode for a short window so it does not
        # immediately collapse when returning from a failed episode.
        self._runner.state = self._SimulationState.INIT
        try:
            self._runner.state_machine()
        except Exception:
            pass

        self._runner.ref_base_lin_vel_H[:] = 0.0
        self._runner.ref_base_ang_yaw_dot = 0.0
        self._runner.pitch_delta = 0.0
        for _ in range(self.post_reset_stand_steps):
            self._advance_once()

        self._runner.state = self._SimulationState.ACTIVE
        try:
            self._runner.state_machine()
        except Exception:
            pass

    def _is_robot_upright_after_reset(self) -> bool:
        try:
            snap = self._runner.mpc_controller.get_rl_state_snapshot()
            if isinstance(snap, dict):
                pos = np.asarray(snap.get('position', np.zeros(3)), dtype=np.float64)
                return bool(float(pos[2]) >= self.reset_recovery_base_z_min)
        except Exception:
            pass
        return True

    def _get_obs(self) -> np.ndarray:
        self._sync_controller_obs_context()
        obs = np.asarray(self._runner.mpc_controller.get_rl_observation_vector(), dtype=np.float32).reshape(-1)
        if obs.shape[0] != int(self.RL_OBS_DIM):
            raise RuntimeError(
                f"Observation dim mismatch: got {obs.shape[0]}, expected {self.RL_OBS_DIM}. "
                "Expected controller_obs(88) with residual_gate, episode_phase, distance_to_course_end, and swing_time."
            )
        return obs

    def _sync_controller_obs_context(self, base_x_override: Optional[float] = None) -> None:
        """Send residual gate and episode context to MPC controller for RL observation assembly."""
        snap = self._runner.mpc_controller.get_rl_state_snapshot()
        if base_x_override is not None:
            base_x = float(base_x_override)
        elif isinstance(snap, dict):
            pos = np.asarray(snap.get('position', np.zeros(3)), dtype=np.float32)
            base_x = float(pos[0]) if pos.size >= 1 else 0.0
        else:
            base_x = 0.0

        episode_phase = float(np.clip(self._episode_step / max(1.0, float(self.episode_length)), 0.0, 1.0))
        distance_to_course_end = float(np.clip(self._course_end_x - base_x, -5.0, 10.0) / 10.0)
        try:
            self._runner.mpc_controller.set_rl_observation_context(
                residual_gate=float(self._last_residual_gate),
                episode_phase=episode_phase,
                distance_to_course_end=distance_to_course_end,
                swing_time=float(getattr(self, '_prev_swing_time_s', self._nominal_swing_time_s)),
            )
        except Exception:
            pass

    def _reward_and_done(self, residual_m: float, swing_time_s: float):
        snap = self._runner.mpc_controller.get_rl_state_snapshot()
        if snap is None:
            return 0.0, False, {
                'tracking_error': 0.0,
                'action_penalty': 0.0,
                'swing_time_action_penalty': 0.0,
                'residual_smoothness_penalty': 0.0,
                'swing_time_smoothness_penalty': 0.0,
            }

        lin_vel = np.asarray(snap.get('linear_velocity', np.zeros(3)), dtype=np.float64)
        ref_lin_vel = np.asarray(snap.get('ref_base_lin_vel', np.zeros(3)), dtype=np.float64)
        ref_lin_vel_effective = np.asarray(snap.get('ref_base_lin_vel_effective', ref_lin_vel), dtype=np.float64)
        euler = np.asarray(snap.get('base_ori_euler_xyz', np.zeros(3)), dtype=np.float64)
        pos = np.asarray(snap.get('position', np.zeros(3)), dtype=np.float64)

        vx = float(lin_vel[0])
        roll = float(euler[0])
        pitch = float(euler[1])
        yaw = float(euler[2])
        z = float(pos[2])
        
        # Track previous residual for smoothness penalty
        if not hasattr(self, '_prev_residual_m'):
            self._prev_residual_m = 0.0
        if not hasattr(self, '_prev_swing_time_s'):
            self._prev_swing_time_s = float(self._nominal_swing_time_s)
        current_residual_m = float(residual_m)
        current_swing_time_s = float(swing_time_s)

        if self._episode_start_yaw is None:
            self._episode_start_yaw = float(yaw)
        yaw_delta = float(yaw - float(self._episode_start_yaw))
        yaw_error = float(np.arctan2(np.sin(yaw_delta), np.cos(yaw_delta)))
        self._episode_max_abs_yaw_error = max(self._episode_max_abs_yaw_error, abs(yaw_error))

        # Track forward velocity against commanded forward velocity.
        commanded_speed_nominal = float(np.linalg.norm(ref_lin_vel[0:2]))
        actual_speed_nominal = float(np.linalg.norm(lin_vel[0:2]))
        nominal_cmd_vx = float(ref_lin_vel[0])
        effective_cmd_vx = float(ref_lin_vel_effective[0])
        sigma_vx = 0.25  # paper value
        sigma_yaw = 0.25  # paper value
        # Avoid division by zero if commanded speed is extremely low
        cmd_vx_safe = max(effective_cmd_vx, 0.1)
        progress_reward = float(np.clip(vx, 0.0, cmd_vx_safe) / cmd_vx_safe)
        tracking_error = abs(vx - effective_cmd_vx)
        forward_tracking_reward = 40 * float(np.exp(-((tracking_error * tracking_error) / sigma_vx))) * progress_reward
        action_penalty = -10.0 * ((current_residual_m / max(self._max_residual_m, 1e-6)) ** 2)
        swing_time_half_range = max(0.5 * (self._max_swing_time_s - self._min_swing_time_s), 1e-6)
        swing_time_action_penalty = -10.0 * (((current_swing_time_s - self._nominal_swing_time_s) / swing_time_half_range) ** 2)
        
        # NEW: Reward residual smoothness (penalize jerky changes)
        residual_smoothness_penalty = -500.0 * ((current_residual_m - self._prev_residual_m) ** 2)
        self._prev_residual_m = float(current_residual_m)

        swing_time_smoothness_penalty = -500.0 * ((current_swing_time_s - self._prev_swing_time_s) ** 2)
        self._prev_swing_time_s = float(current_swing_time_s)

        overspeed_limit = max(self.overspeed_terminate_mps, 2.5 * abs(self._episode_command_forward) + 0.30)
        # Calculate how high the robot is above nominal walking height
        height_above_ground = np.clip(z - 0.30, 0.0, 0.35)
        # Only grant the altitude bonus if the robot is actually walking forward
        on_stairs = self.stair_start_x - 0.5 <= base_x <= self._course_end_x
        climb_bonus = 10.0 * height_above_ground * progress_reward if on_stairs else 0.0

        # Time bonus is now applied as a one-time terminal bonus to prevent per-step farming,
        # but we compute a value for information logging if needed.
        time_bonus = 0.0

        # Penalize lateral velocity tracking error instead of lateral position.
        # This aligns the penalty with commanded lateral motion (typically 0 m/s).
        lateral_y = float(pos[1])
        actual_lateral_vy = float(lin_vel[1])
        commanded_lateral_vy = float(ref_lin_vel_effective[1])
        lateral_velocity_error = float(actual_lateral_vy - commanded_lateral_vy)
        
        # IMPROVED: Phase-gated lateral penalty
        # Reduce lateral penalty during stair interaction (within ±0.5m of stair center)
        stair_interaction_zone = abs(pos[0] - self.stair_start_x - self._stair_num_steps_up * self._stair_step_width / 2) < 1.0
        if stair_interaction_zone:
            lateral_deviation_penalty = -0.15 * (lateral_velocity_error * lateral_velocity_error)  # Reduced during stair interaction
        else:
            lateral_deviation_penalty = -0.8 * (lateral_velocity_error * lateral_velocity_error)  # Normal penalty on flat terrain
        
        # NEW: Roll/pitch angular velocity penalties for stability
        # Penalize excessive roll and pitch rotation rates to stabilize base attitude
        ang_vel = np.asarray(snap.get('base_ang_vel_body_xyz', np.zeros(3)), dtype=np.float64)
        roll_rate = float(ang_vel[0])
        pitch_rate = float(ang_vel[1])
        yaw_rate = float(ang_vel[2])  # Track for completeness
        commanded_yaw_rate = float(snap.get('ref_base_ang_yaw_dot', getattr(self._runner, 'ref_base_ang_yaw_dot', 0.0)))
        yaw_rate_error = float(yaw_rate - commanded_yaw_rate)
        yaw_tracking_reward = 10 * float(np.exp(-((yaw_rate_error * yaw_rate_error) / sigma_yaw))) * progress_reward
        attitude_rate_penalty = -2.0 * (roll_rate**2 + pitch_rate**2)  # Penalize fast rotations around X and Y
        orientation_penalty = -5.0 * (roll * roll + pitch * pitch)

        # Penalize persistently weak forward progress once out of startup grace.
        slow_progress_penalty = 0.0
        if self._episode_step > max(self.termination_grace_steps, 40):
            min_progress_vx = max(0.15, 0.30 * abs(self._episode_command_forward))
            if vx < min_progress_vx:
                slow_progress_penalty = -0.2 * float((min_progress_vx - vx) ** 2)
        
        # NEW: Torque saturation cost from MPC diagnostics
        # Penalize episodes where MPC saturates torques frequently (indicates control difficulty)
        mpc_debug = getattr(self._runner, 'last_mpc_debug_info', {})
        mpc_diag = mpc_debug.get('mpc_diag', {}) if isinstance(mpc_debug, dict) else {}
        torque_sat_ratio = float(mpc_diag.get('torque_sat_ratio', 0.0)) if isinstance(mpc_diag, dict) else 0.0
        torque_saturation_penalty = -15.0 * torque_sat_ratio  # Stronger penalty to discourage saturated MPC solutions
        
        reward = (
            forward_tracking_reward
            + yaw_tracking_reward
            + progress_reward
            + action_penalty
            + swing_time_action_penalty
            + residual_smoothness_penalty   #NEW: penalizes jerky residual changes between steps
            + swing_time_smoothness_penalty
            + float(climb_bonus)    #rewards the robot for increasing base height
            + float(lateral_deviation_penalty) #penalizes lateral deviation from the centerline, quadratic penalty on the lateral position
            + float(orientation_penalty)
            + float(time_bonus) #rewards faster course completion
            + float(attitude_rate_penalty)  # NEW: penalizes fast roll/pitch changes
            + float(torque_saturation_penalty)  # NEW: penalizes MPC torque saturation
            + float(slow_progress_penalty)
        )

        # IMPROVED: Reduced reward clipping asymmetry from [-20, 100] to [-30, 100] to preserve more information about bad episodes
        # This allows bad episodes to still be informative without dominating the value function.
        reward = float(np.clip(reward, -30.0, 100.0))

        in_grace = self._episode_step < self.termination_grace_steps
        tilt_violation = (abs(roll) > 1.2) or (abs(pitch) > 1.2)
        hard_tilt_violation = (abs(roll) > self.hard_attitude_terminate_rad) or (abs(pitch) > self.hard_attitude_terminate_rad)
        overspeed_violation = abs(vx) > (overspeed_limit + 0.35)

        if tilt_violation:
            self._tilt_violation_count += 1
        else:
            self._tilt_violation_count = 0

        if overspeed_violation:
            self._overspeed_violation_count += 1
        else:
            self._overspeed_violation_count = 0

        if hard_tilt_violation:
            self._hard_tilt_violation_count += 1
        else:
            self._hard_tilt_violation_count = 0

        termination_armed = self._last_residual_gate >= 0.99

        if in_grace:
            terminated = bool(
                z < float(self.base_height_grace)
                or self._hard_tilt_violation_count >= self.hard_attitude_violation_steps
            )
            if z < float(self.base_height_grace):
                termination_reason = 'base_height_grace'
            elif self._hard_tilt_violation_count >= self.hard_attitude_violation_steps:
                termination_reason = 'attitude_hard'
            else:
                termination_reason = 'none'
        else:
            terminated = bool(
                z < float(self.base_height_terminate)
                or self._hard_tilt_violation_count >= self.hard_attitude_violation_steps
                or (
                    termination_armed
                    and self._tilt_violation_count >= self.attitude_violation_steps
                )
                or (
                    termination_armed
                    and self._overspeed_violation_count >= self.overspeed_violation_steps
                )
            )
            if z < float(self.base_height_terminate):
                termination_reason = 'base_height'
            elif self._hard_tilt_violation_count >= self.hard_attitude_violation_steps:
                termination_reason = 'attitude_hard'
            elif termination_armed and self._tilt_violation_count >= self.attitude_violation_steps:
                termination_reason = 'attitude'
            elif termination_armed and self._overspeed_violation_count >= self.overspeed_violation_steps:
                termination_reason = 'overspeed'
            else:
                termination_reason = 'none'

        info = {
            'tracking_error': float(tracking_error),
            'forward_tracking_reward': float(forward_tracking_reward),
            'yaw_tracking_reward': float(yaw_tracking_reward),
            'yaw_rate_error': float(yaw_rate_error),
            'action_penalty': float(action_penalty),
            'swing_time_action_penalty': float(swing_time_action_penalty),
            'residual_smoothness_penalty': float(residual_smoothness_penalty),  # NEW
            'swing_time_smoothness_penalty': float(swing_time_smoothness_penalty),
            'climb_bonus': float(climb_bonus),
            'lateral_deviation_penalty': float(lateral_deviation_penalty),
            'orientation_penalty': float(orientation_penalty),
            'attitude_rate_penalty': float(attitude_rate_penalty),  # NEW
            'torque_saturation_penalty': float(torque_saturation_penalty),  # NEW
            'slow_progress_penalty': float(slow_progress_penalty),
            'roll_rate': float(roll_rate),  # NEW: for debugging
            'pitch_rate': float(pitch_rate),  # NEW: for debugging
            'yaw_rate': float(yaw_rate),  # NEW: for debugging
            'torque_sat_ratio': float(torque_sat_ratio),  # NEW: MPC diagnostic
            'lateral_velocity_error': float(lateral_velocity_error),
            'actual_lateral_vy': float(actual_lateral_vy),
            'cmd_lateral_vy': float(commanded_lateral_vy),
            'yaw_error': float(yaw_error),
            'time_bonus': float(time_bonus),
            'base_y': float(lateral_y),
            'base_yaw': float(yaw),
            'base_roll': float(roll),
            'base_pitch': float(pitch),
            'overspeed_limit': float(overspeed_limit),
            'vx': float(vx),
            'actual_speed_nominal': float(actual_speed_nominal),
            'cmd_speed_nominal': float(commanded_speed_nominal),
            'cmd_vx': float(nominal_cmd_vx),
            'cmd_vx_nominal': float(nominal_cmd_vx),
            'cmd_vx_effective': float(effective_cmd_vx),
            'base_z': float(z),
            'base_x': float(pos[0]),
            'step_height_effective': float(snap.get('step_height_effective', 0.0)),
            'termination_reason': termination_reason,
            'in_termination_grace': float(1.0 if in_grace else 0.0),
            'termination_armed': float(1.0 if termination_armed else 0.0),
            'overspeed_termination_armed': float(1.0 if termination_armed else 0.0),
            'attitude_termination_armed': float(1.0 if termination_armed else 0.0),
            'tilt_violation_count': float(self._tilt_violation_count),
            'hard_tilt_violation_count': float(self._hard_tilt_violation_count),
            'overspeed_violation_count': float(self._overspeed_violation_count),
            'hard_attitude_threshold_rad': float(self.hard_attitude_terminate_rad),
            'base_height_terminate': float(self.base_height_terminate),
            'base_height_grace': float(self.base_height_grace),
        }
        try:
            mpc_debug = getattr(self._runner, 'last_mpc_debug_info', {})
            mpc_diag = mpc_debug.get('mpc_diag', {}) if isinstance(mpc_debug, dict) else {}
            if isinstance(mpc_diag, dict):
                info['mpc_status'] = float(mpc_diag.get('solve_status', -999))
                info['mpc_compute_ms'] = float(mpc_diag.get('compute_duration_ms', 0.0))
                info['mpc_torque_max_abs'] = float(mpc_diag.get('torque_max_abs', 0.0))
                info['mpc_torque_sat_ratio'] = float(mpc_diag.get('torque_sat_ratio', 0.0))
                info['mpc_sqp_iter'] = float(mpc_diag.get('sqp_iter', -1))
                info['mpc_qp_iter'] = float(mpc_diag.get('qp_iter', -1))
                info['mpc_qp_status'] = float(mpc_diag.get('qp_status', -1))
                info['mpc_maxiter_count_episode'] = float(mpc_diag.get('acados_maxiter_count_episode', 0.0))
                info['mpc_minstep_count_episode'] = float(mpc_diag.get('acados_minstep_count_episode', 0.0))
                info['mpc_failure_count_episode'] = float(mpc_diag.get('acados_failure_count_episode', 0.0))
        except Exception:
            pass
        return float(reward), terminated, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if self.fixed_course and bool(self._world.params.add_stairs):
            # Always build the same fixed obstacle sets (idempotent)
            self._apply_obstacle_sets(self._build_fixed_obstacle_sets())
        elif self.randomize_stairs and bool(self._world.params.add_stairs):
            self._apply_stair_geometry(self._sample_stair_geometry())
        elif self.multi_obstacles_enabled and bool(self._world.params.add_stairs):
            self._apply_obstacle_sets(self._sample_obstacle_sets())
        self._world.world.reset()
        self._episode_count += 1
        self._episode_command_forward = float(self._sample_episode_command_forward())

        self._episode_reward_comp_sums = {
            'base_reward': 0.0,
            'forward_tracking_reward': 0.0,
            'yaw_tracking_reward': 0.0,
            'tracking_term': 0.0,
            'action_penalty': 0.0,
            'swing_time_action_penalty': 0.0,
            'residual_smoothness_penalty': 0.0,
            'swing_time_smoothness_penalty': 0.0,
            'climb_bonus': 0.0,
            'lateral_deviation_penalty': 0.0,
            'orientation_penalty': 0.0,
            'time_bonus': 0.0,
            'total_reward': 0.0,
            'attitude_rate_penalty': 0.0,
            'torque_saturation_penalty': 0.0,
            'slow_progress_penalty': 0.0,
            'course_complete_terminal_bonus': 0.0,
            'attitude_hard_terminal_penalty': 0.0,
            'stagnation_terminal_penalty': 0.0,
        }

        try:
            self._runner.hard_reset_robot_state(settle_steps=20)
        except Exception:
            pass

        # Let the robot stabilise in INIT/standing mode briefly so it is upright
        # before MPC takes over.  This uses initial_dof_props (with damping) so
        # the standing controller can hold the pose properly.
        self._post_reset_recovery()

        # If the robot is still collapsed after recovery, retry once.
        if not self._is_robot_upright_after_reset():
            try:
                self._runner.hard_reset_robot_state(settle_steps=30)
            except Exception:
                pass
            self._post_reset_recovery()

        # Reset runner/controller episode-scoped state so startup protections and
        # MPC warm-start state do not leak across episodes.
        try:
            self._runner.physics_step_counter = 0
            self._runner.last_torques = None
            self._runner.last_mpc_debug_info = {}
        except Exception:
            pass
        try:
            self._runner.mpc_controller.reset_for_new_episode()
        except Exception:
            pass
        self._runner.mpc_controller.enable_rl_step_height(True)
        self._runner.mpc_controller.set_step_height_residual(0.0)
        try:
            self._runner.mpc_controller.set_adaptive_swing_time(float(self._nominal_swing_time_s))
        except Exception:
            pass
        self._episode_step = 0
        self._episode_max_x = 0.0
        self._episode_max_z = 0.0
        self._episode_start_z = 0.0
        self._episode_climbed_peak = False
        self._episode_descended_to_flat = False
        self._success_post_steps = 0
        self._last_residual_gate = 0.0
        self._prev_residual_m = 0.0
        self._prev_swing_time_s = float(self._nominal_swing_time_s)
        self._tilt_violation_count = 0
        self._hard_tilt_violation_count = 0
        self._overspeed_violation_count = 0
        self._course_entry_step = -1
        self._course_complete_step = -1
        self._stagnation_x_history = deque(maxlen=self.stagnation_window_steps)
        self._episode_start_yaw = None
        self._episode_max_abs_yaw_error = 0.0

        self._set_command()
        for _ in range(8):
            self._advance_once()

        try:
            snap = self._runner.mpc_controller.get_rl_state_snapshot()
            if isinstance(snap, dict):
                pos = np.asarray(snap.get('position', np.zeros(3)), dtype=np.float64)
                euler = np.asarray(snap.get('base_ori_euler_xyz', np.zeros(3)), dtype=np.float64)
                self._episode_start_z = float(pos[2])
                self._episode_start_yaw = float(euler[2])
        except Exception:
            self._episode_start_z = 0.0
            self._episode_start_yaw = None

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        # Keep obstacle retry and reflex disabled in both phases so RL residual
        # is solely responsible for step-height adaptation.
        self._cfg.simulation_params['obstacle_retry_enabled'] = False
        self._cfg.simulation_params['reflex_trigger_mode'] = False
        self._cfg.simulation_params['reflex_next_steps_height_enhancement'] = False

        # Reflect phase policy in live detector objects as they cache settings at
        # construction time and do not automatically re-read cfg on every step.
        try:
            esd = self._runner.mpc_controller.wb_interface.esd
            esd.trigger_mode = self._cfg.simulation_params['reflex_trigger_mode']
            esd.activated = bool(esd.trigger_mode is not False)
            esd.use_reflex_next_steps_height_enhancement = bool(
                self._cfg.simulation_params['reflex_next_steps_height_enhancement']
            )
        except Exception:
            pass

        if action.size < 1:
            raise ValueError("Expected at least one action component for step-height residual")
        a_residual = float(np.clip(action[0], -1.0, 1.0))
        a_swing_time = float(np.clip(action[1], -1.0, 1.0)) if action.size > 1 else 0.0
        gate = 1.0
        if self._episode_step < self.residual_warmup_steps:
            gate = 0.0
        elif self.residual_ramp_steps > 0:
            gate = np.clip(
                (self._episode_step - self.residual_warmup_steps) / float(self.residual_ramp_steps),
                0.0,
                1.0,
            )
        self._last_residual_gate = float(gate)
        residual_m = a_residual * self._max_residual_m * float(gate)
        target_swing_time_s = float(
            self._min_swing_time_s + 0.5 * (a_swing_time + 1.0) * (self._max_swing_time_s - self._min_swing_time_s)
        )
        swing_time_s = float(self._nominal_swing_time_s + float(gate) * (target_swing_time_s - self._nominal_swing_time_s))

        self._runner.mpc_controller.set_step_height_residual(residual_m)
        try:
            self._runner.mpc_controller.set_adaptive_swing_time(swing_time_s)
        except Exception:
            pass
        self._set_command()

        reward_sum = 0.0
        terminated = False
        last_info = {}
        if not hasattr(self, '_episode_reward_comp_sums'):
            self._episode_reward_comp_sums = {k: 0.0 for k in [
                'base_reward', 'forward_tracking_reward', 'yaw_tracking_reward',
                'tracking_term', 'action_penalty', 'swing_time_action_penalty',
                'residual_smoothness_penalty', 'swing_time_smoothness_penalty',
                'climb_bonus', 'lateral_deviation_penalty', 'orientation_penalty',
                'time_bonus', 'total_reward',
                'attitude_rate_penalty', 'torque_saturation_penalty',
                'slow_progress_penalty', 'course_complete_terminal_bonus',
                'attitude_hard_terminal_penalty', 'stagnation_terminal_penalty'
            ]}
        reward_component_sums = self._episode_reward_comp_sums

        for _ in range(self.frame_skip):
            self._advance_once()
            r, done, info = self._reward_and_done(residual_m, swing_time_s)
            reward_sum += float(r)
            last_info = info
            last_info['residual_m'] = float(residual_m)
            last_info['swing_time_s'] = float(swing_time_s)
            last_info['residual_gate'] = float(gate)
            # Aggregate true per-component sums over frame_skip so trainer can
            # log episode-level decomposition to TensorBoard.
            reward_component_sums['base_reward'] += 0.25
            reward_component_sums['forward_tracking_reward'] += float(info.get('forward_tracking_reward', 0.0))
            reward_component_sums['yaw_tracking_reward'] += float(info.get('yaw_tracking_reward', 0.0))
            reward_component_sums['tracking_term'] += -float(info.get('tracking_error', 0.0))
            reward_component_sums['action_penalty'] += float(info.get('action_penalty', 0.0))
            reward_component_sums['swing_time_action_penalty'] += float(info.get('swing_time_action_penalty', 0.0))
            reward_component_sums['residual_smoothness_penalty'] += float(info.get('residual_smoothness_penalty', 0.0))  # NEW
            reward_component_sums['swing_time_smoothness_penalty'] += float(info.get('swing_time_smoothness_penalty', 0.0))
            reward_component_sums['climb_bonus'] += float(info.get('climb_bonus', 0.0))
            reward_component_sums['lateral_deviation_penalty'] += float(info.get('lateral_deviation_penalty', 0.0))
            reward_component_sums['orientation_penalty'] += float(info.get('orientation_penalty', 0.0))
            reward_component_sums['time_bonus'] += float(info.get('time_bonus', 0.0))
            reward_component_sums['attitude_rate_penalty'] += float(info.get('attitude_rate_penalty', 0.0))  # NEW
            reward_component_sums['torque_saturation_penalty'] += float(info.get('torque_saturation_penalty', 0.0))  # NEW
            reward_component_sums['slow_progress_penalty'] += float(info.get('slow_progress_penalty', 0.0))
            reward_component_sums['total_reward'] += float(r)

            base_x = float(info.get('base_x', 0.0))
            base_z = float(info.get('base_z', 0.0))
            self._episode_max_x = max(self._episode_max_x, base_x)
            self._episode_max_z = max(self._episode_max_z, base_z)

            # Completion time: record when robot enters the course
            if self._course_entry_step < 0 and base_x >= self.stair_start_x:
                self._course_entry_step = self._episode_step

            if (base_x >= (self._stair_up_end_x - 0.5 * self._stair_step_width)) and (base_z >= (self._stair_top_height - 0.03)):
                self._episode_climbed_peak = True

            # Deviation termination: robot has moved far past stair start but
            # has never climbed the stairs (typically side bypass/deviation).
            base_y = float(info.get('base_y', 0.0))
            forward_bypass = (base_x - self.stair_start_x) >= self.deviation_no_stair_distance_m
            lateral_bypass = (
                (base_x >= (self.stair_start_x - self.deviation_lateral_activation_x_margin_m))
                and (abs(base_y) >= self.deviation_lateral_no_stair_m)
            )
            if (not self._episode_climbed_peak) and (forward_bypass or lateral_bypass):
                terminated = True
                last_info['termination_reason'] = 'deviation'
                last_info['deviation_forward_bypass'] = float(1.0 if forward_bypass else 0.0)
                last_info['deviation_lateral_bypass'] = float(1.0 if lateral_bypass else 0.0)
                last_info['deviation_lateral_threshold_m'] = float(self.deviation_lateral_no_stair_m)
                done = True

            # Simplified descent: success = passed course end AND near flat ground
            if self._episode_climbed_peak:
                passed_course_end = base_x >= (self._course_end_x - self._descend_down_end_x_margin_m)
                near_flat = base_z <= (self._episode_start_z + self._descend_flat_z_margin_m)
                if passed_course_end and near_flat:
                    self._episode_descended_to_flat = True
                    # Record completion time on first success
                    if self._course_complete_step < 0:
                        self._course_complete_step = self._episode_step

            if self.success_reset_enabled:
                passed_goal_x = base_x >= (self._course_end_x + self.success_reset_x_margin_m)
                if self._episode_descended_to_flat and passed_goal_x and (self._episode_step >= self.success_reset_min_episode_step):
                    self._success_post_steps += 1
                else:
                    self._success_post_steps = 0

                if self._success_post_steps >= self.success_reset_hold_steps:
                    terminated = True
                    last_info['termination_reason'] = 'course_complete'
                    done = True

            if done:
                terminated = True
                break

        self._sync_controller_obs_context(base_x_override=float(last_info.get('base_x', 0.0)))

        self._episode_step += 1
        self._cumulative_env_steps += 1  # Track cumulative steps for learning_starts gating

        # Stagnation check: terminate if robot hasn't advanced enough.
        # The gate requirement is intentionally omitted so that a stuck robot is
        # terminated early even during the residual warmup/ramp window.
        base_x_now = float(last_info.get('base_x', 0.0))
        self._stagnation_x_history.append(base_x_now)
        if (len(self._stagnation_x_history) >= self.stagnation_window_steps
                and self._episode_step > self.termination_grace_steps + self.stagnation_window_steps):
            oldest_x = self._stagnation_x_history[0]
            if (base_x_now - oldest_x) < self.stagnation_threshold_m:
                terminated = True
                last_info['termination_reason'] = 'stagnation'

        truncated = bool(self._episode_step >= self.episode_length)

        # Apply one-time terminal shaping aligned with the true task objective.
        terminal_reason = str(last_info.get('termination_reason', 'none'))
        if terminated:
            if terminal_reason == 'course_complete':
                reward_sum += float(self.course_complete_terminal_bonus)
                reward_component_sums['course_complete_terminal_bonus'] += float(self.course_complete_terminal_bonus)
                
                # Apply terminal time bonus based on completion speed
                if self._course_entry_step >= 0 and self._course_complete_step > 0:
                    completion_steps = int(self._course_complete_step - self._course_entry_step)
                    time_bonus_terminal = 2000.0 * float(np.clip(1.0 - completion_steps / 5000.0, 0.0, 1.0))
                    reward_sum += time_bonus_terminal
                    reward_component_sums['time_bonus'] += time_bonus_terminal
                    
            elif terminal_reason == 'attitude_hard':
                reward_sum -= float(self.attitude_hard_terminal_penalty)
                reward_component_sums['attitude_hard_terminal_penalty'] -= float(self.attitude_hard_terminal_penalty)
            elif terminal_reason == 'stagnation':
                reward_sum -= float(self.stagnation_terminal_penalty)
                reward_component_sums['stagnation_terminal_penalty'] -= float(self.stagnation_terminal_penalty)

        for key, value in reward_component_sums.items():
            last_info[f'reward_comp_sum_{key}'] = float(value)

        last_info['course_complete_terminal_bonus'] = float(reward_component_sums['course_complete_terminal_bonus'])
        last_info['attitude_hard_terminal_penalty'] = float(reward_component_sums['attitude_hard_terminal_penalty'])
        last_info['stagnation_terminal_penalty'] = float(reward_component_sums['stagnation_terminal_penalty'])

        if terminated or truncated:
            reached_stairs = bool(self._episode_max_x >= (self.stair_start_x - 0.10))
            climbed_up = bool(self._episode_climbed_peak)
            descended_to_flat = bool(self._episode_descended_to_flat)
            course_success = bool(descended_to_flat)
            course_complete = bool(str(last_info.get('termination_reason', 'none')) == 'course_complete')
            progression_success = course_complete if self.curriculum_progression_criterion == 'course_complete' else course_success
            self._update_curriculum_progress(progression_success)

            # Completion time: -1 means not completed
            if self._course_entry_step >= 0 and self._course_complete_step > 0:
                completion_steps = int(self._course_complete_step - self._course_entry_step)
            else:
                completion_steps = -1

            if self.speed_curriculum_enabled:
                if self.curriculum_gate_enabled:
                    curriculum_progress = float(self._curriculum_progress)
                elif self._episode_count <= self.speed_curriculum_warmup_episodes:
                    curriculum_progress = 0.0
                elif self._episode_count >= self.speed_curriculum_full_episodes:
                    curriculum_progress = 1.0
                else:
                    denom = float(self.speed_curriculum_full_episodes - self.speed_curriculum_warmup_episodes)
                    curriculum_progress = float((self._episode_count - self.speed_curriculum_warmup_episodes) / max(denom, 1.0))
                curriculum_progress = float(np.clip(curriculum_progress, 0.0, 1.0))
                curriculum_current_max = float(
                    np.clip(
                        self.speed_curriculum_min + curriculum_progress * (self.speed_curriculum_max - self.speed_curriculum_min),
                        self.speed_curriculum_min,
                        self.speed_curriculum_max,
                    )
                )
            else:
                curriculum_progress = 1.0
                curriculum_current_max = float(self.command_forward)

            last_info = dict(last_info)
            last_info.update({
                'episode_max_x': float(self._episode_max_x),
                'episode_max_z': float(self._episode_max_z),
                'episode_start_z': float(self._episode_start_z),
                'episode_residual_gate_final': float(self._last_residual_gate),
                'episode_swing_time_final': float(getattr(self._runner.mpc_controller.wb_interface, 'adaptive_swing_time', self._nominal_swing_time_s)),
                'episode_reached_stairs': float(1.0 if reached_stairs else 0.0),
                'episode_climbed_up': float(1.0 if climbed_up else 0.0),
                'episode_descended_to_flat': float(1.0 if descended_to_flat else 0.0),
                'episode_course_success': float(1.0 if course_success else 0.0),
                'episode_course_complete': float(1.0 if course_complete else 0.0),
                'episode_curriculum_progression_success': float(1.0 if progression_success else 0.0),
                'episode_curriculum_progression_criterion_is_complete': float(1.0 if self.curriculum_progression_criterion == 'course_complete' else 0.0),
                'episode_completion_steps': float(completion_steps),
                'episode_time_to_complete_s': float(completion_steps * self._runner.sim_dt) if completion_steps > 0 else -1.0,
                'episode_course_length': float(self._course_length),
                'episode_course_end_x': float(self._course_end_x),
                'episode_max_abs_yaw_error': float(self._episode_max_abs_yaw_error),
                'episode_end_reason': 'terminated' if terminated else 'truncated',
                'episode_termination_reason': str(last_info.get('termination_reason', 'none')),
                'episode_mpc_maxiter_count': float(last_info.get('mpc_maxiter_count_episode', 0.0)),
                'episode_mpc_minstep_count': float(last_info.get('mpc_minstep_count_episode', 0.0)),
                'episode_mpc_failure_count': float(last_info.get('mpc_failure_count_episode', 0.0)),
                'episode_command_forward': float(self._episode_command_forward),
                'episode_curriculum_min': float(self.speed_curriculum_min),
                'episode_curriculum_max': float(self.speed_curriculum_max),
                'episode_curriculum_progress': float(curriculum_progress),
                'episode_curriculum_current_max': float(curriculum_current_max),
                'episode_curriculum_stage': float(self._curriculum_stage),
                'episode_curriculum_two_stage_success_rate': float(
                    np.mean(self._curriculum_stage_success) if len(self._curriculum_stage_success) > 0 else 0.0
                ),
                'episode_curriculum_recent_success_rate': float(np.mean(self._recent_success) if len(self._recent_success) > 0 else 0.0),
                'episode_curriculum_success_threshold_effective': float(self.curriculum_success_threshold),
                'episode_curriculum_window_episodes_effective': float(self.curriculum_window_episodes),
                'episode_stair_start_x': float(self.stair_start_x),
                'episode_stair_step_height': float(self._stair_step_height),
                'episode_stair_step_width': float(self._stair_step_width),
                'episode_stair_num_steps_up': float(self._stair_num_steps_up),
                'episode_stair_num_steps_down': float(self._stair_num_steps_down),
                'episode_extra_obstacle_sets': float(self.extra_obstacle_sets if self.multi_obstacles_enabled else 0),
            })

        obs = self._get_obs()
        return obs, float(reward_sum), terminated, truncated, last_info

    def close(self):
        try:
            if hasattr(self, '_cfg') and hasattr(self, '_orig_obstacle_retry_enabled'):
                self._cfg.simulation_params['obstacle_retry_enabled'] = bool(self._orig_obstacle_retry_enabled)
            if hasattr(self, '_cfg') and hasattr(self, '_orig_reflex_trigger_mode'):
                self._cfg.simulation_params['reflex_trigger_mode'] = self._orig_reflex_trigger_mode
            if hasattr(self, '_cfg') and hasattr(self, '_orig_reflex_next_steps_height_enhancement'):
                self._cfg.simulation_params['reflex_next_steps_height_enhancement'] = bool(
                    self._orig_reflex_next_steps_height_enhancement
                )
        except Exception:
            pass
        try:
            if hasattr(self, '_simulation_app') and self._simulation_app is not None:
                self._simulation_app.close()
        except Exception:
            pass

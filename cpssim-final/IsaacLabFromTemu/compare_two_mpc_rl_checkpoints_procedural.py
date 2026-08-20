from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import queue
import random
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


_REQUIRED_CASADI_LIBDIR = "/home/mena/miniconda3/envs/pympc/lib"
_ld = os.environ.get("LD_LIBRARY_PATH", "")
_ld_parts = [part for part in _ld.split(":") if part]
if _REQUIRED_CASADI_LIBDIR not in _ld_parts:
    new_env = dict(os.environ)
    new_env["LD_LIBRARY_PATH"] = ":".join([_REQUIRED_CASADI_LIBDIR] + _ld_parts)
    os.execvpe(sys.executable, [sys.executable] + sys.argv, new_env)


def _ensure_isaac_sim_python_path() -> None:
    """Prioritize Isaac Sim site-packages for worker interpreter imports."""
    import glob

    isaac_paths = [
        "/home/mena/isaacsim/kit/python/lib/python3.11/site-packages",
        "/home/mena/isaacsim/exts",
        "/home/mena/isaacsim/extsDeprecated",
        "/home/mena/isaacsim/extsDeprecated/omni.isaac.dynamic_control",
    ]
    isaac_paths.extend(
        glob.glob("/home/mena/isaacsim/exts/isaacsim.*")
    )
    isaac_paths.extend(
        glob.glob("/home/mena/isaacsim/extscache/omni.kit.numpy.common-*.lx64.r.cp311")
    )
    isaac_paths.extend(
        glob.glob("/home/mena/isaacsim/extscache/omni.kit.commands-*.lx64.r.cp311")
    )
    isaac_paths.extend(
        glob.glob("/home/mena/isaacsim/extscache/omni.kit.actions.core-*.lx64.r.cp311")
    )
    isaac_paths.extend(
        glob.glob("/home/mena/isaacsim/extscache/omni.usd-*.lx64.r.cp311")
    )
    isaac_paths.extend(
        glob.glob("/home/mena/isaacsim/extscache/omni.kit.*")
    )
    isaac_paths.extend(
        glob.glob("/home/mena/isaacsim/extscache/omni.usd.*")
    )
    isaac_paths.extend(
        glob.glob("/home/mena/isaacsim/kit/extscore/*")
    )
    for isaac_path in isaac_paths:
        if os.path.isdir(isaac_path):
            if isaac_path in sys.path:
                sys.path.remove(isaac_path)
            sys.path.insert(0, isaac_path)


_HEADLESS_DEFAULT = True
if "--no-headless" in sys.argv:
    _HEADLESS_DEFAULT = False
elif "--headless" in sys.argv:
    _HEADLESS_DEFAULT = True


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if os.path.join(REPO_ROOT, "pympc") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "pympc"))
if os.path.dirname(__file__) not in sys.path:
    sys.path.insert(0, os.path.dirname(__file__))

from quadruped_pympc import config as cfg
from src.rl_config import RLTrainArgs


@dataclass(frozen=True)
class CaseSpec:
    key: str
    label: str
    color: str
    checkpoint: str


@dataclass
class EvalConfig:
    episodes_per_speed: int = 4
    episode_length_steps: int = 20000
    render_every_n_steps: int = 8

    command_forward: float = 0.8
    command_lateral: float = 0.0
    command_yaw: float = 0.0
    speed_min: float = 0.5
    speed_max: float = 1.0
    speed_step: float = 0.1

    procedural_terrain_length_m: float = float(RLTrainArgs.procedural_terrain_length_m)
    procedural_terrain_width_m: float = float(RLTrainArgs.procedural_terrain_width_m)
    procedural_narrow_success_x: float = float(RLTrainArgs.procedural_narrow_success_x)
    random_uneven_terrain: bool = bool(RLTrainArgs.random_uneven_terrain)
    uneven_patch_height_min: float = float(RLTrainArgs.uneven_patch_height_min)
    uneven_patch_height_max: float = float(RLTrainArgs.uneven_patch_height_max)
    uneven_patches_per_gap_max: int = int(RLTrainArgs.uneven_patches_per_gap_max)
    random_slope_terrain: bool = bool(RLTrainArgs.random_slope_terrain)
    slope_angle_deg_min: float = float(RLTrainArgs.slope_angle_deg_min)
    slope_angle_deg_max: float = float(RLTrainArgs.slope_angle_deg_max)
    slope_step_resolution_m: float = float(RLTrainArgs.slope_step_resolution_m)

    survival_success_duration_steps: int = int(RLTrainArgs.survival_success_duration_steps)
    success_reset_hold_steps: int = 20
    success_reset_min_episode_step: int = 120
    success_reset_x_margin_m: float = 0.10

    stagnation_check_window_steps: int = 500
    stagnation_distance_threshold: float = 0.05
    post_reset_stand_steps: int = 15
    command_ramp_steps: int = 40


@dataclass
class EpisodeMetrics:
    terrain: str
    terrain_seed: int
    case: str
    checkpoint_label: str
    episode_idx: int
    command_forward: float
    course_success: float
    course_complete: float
    goal_crossed: float
    completion_time_s: float
    strict_completion_time_s: float
    mean_tracking_error: float
    mean_cmd_speed_nominal: float
    mean_actual_speed_nominal: float
    mean_swing_time_s: float
    mean_cot: float
    mean_step_height_residual: float
    mean_base_x: float
    mean_base_z: float
    mean_swing_foot_z: float
    mean_joint_torque_abs: float
    mean_roll_error_rad: float
    mean_pitch_error_rad: float
    mean_yaw_error_rad: float
    total_steps: int



def _resolve_checkpoint_path(path: str) -> str:
    ckpt = os.path.expanduser(str(path).strip())
    if not ckpt:
        return ""
    if os.path.isabs(ckpt):
        return os.path.abspath(ckpt)
    candidate_repo = os.path.abspath(os.path.join(REPO_ROOT, ckpt))
    if os.path.exists(candidate_repo):
        return candidate_repo
    candidate_cwd = os.path.abspath(ckpt)
    return candidate_cwd



def _build_speed_sweep(speed_min: float, speed_max: float, speed_step: float) -> List[float]:
    if speed_step <= 0:
        raise ValueError("speed_step must be > 0")
    if speed_max < speed_min:
        raise ValueError("speed_max must be >= speed_min")
    values: List[float] = []
    current = float(speed_min)
    while current <= float(speed_max) + 1e-9:
        values.append(round(float(current), 10))
        current += float(speed_step)
    return values



def _safe_nanmean(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))



def _remove_prim_if_present(world, prim_path: str) -> None:
    prim = world.stage.GetPrimAtPath(prim_path)
    if prim and prim.IsValid():
        world.stage.RemovePrim(prim_path)



def _add_procedural_terrain_to_world(world, cfg_eval: EvalConfig, seed: Optional[int] = None) -> None:
    from isaacsim.core.api.materials import PhysicsMaterial
    from isaacsim.core.api.objects import FixedCuboid
    from pxr import PhysxSchema, UsdLux, UsdPhysics

    isaaclab_src = "/home/mena/IsaacLab/source/isaaclab"
    if os.path.isdir(isaaclab_src) and isaaclab_src not in sys.path:
        sys.path.insert(0, isaaclab_src)

    from isaaclab.terrains import TerrainGenerator, TerrainGeneratorCfg, create_prim_from_mesh
    from isaaclab.terrains.height_field import (
        HfDiscreteObstaclesTerrainCfg,
        HfInvertedPyramidStairsTerrainCfg,
        HfPyramidSlopedTerrainCfg,
        HfPyramidStairsTerrainCfg,
        HfRandomUniformTerrainCfg,
    )

    _remove_prim_if_present(world, "/World/random_terrain")
    if not bool(getattr(getattr(world, "params", None), "add_ground_plane", False)):
        _remove_prim_if_present(world, "/World/defaultGroundPlane")

    stage = world.stage

    stair_material_path = "/World/Physics_Materials/StairMaterial"
    if not stage.GetPrimAtPath(stair_material_path):
        stair_material = PhysicsMaterial(
            stair_material_path,
            static_friction=0.6,
            dynamic_friction=0.6,
            restitution=0.0,
        )
        stair_material_prim = stage.GetPrimAtPath(stair_material_path)
        PhysxSchema.PhysxMaterialAPI.Apply(stair_material_prim).CreateFrictionCombineModeAttr().Set("max")
    else:
        stair_material = PhysicsMaterial(stair_material_path)

    if not stage.GetPrimAtPath("/World/random_terrain"):
        stage.DefinePrim("/World/random_terrain", "Xform")

    platform_length = 2.0
    platform_width = 2.0
    platform_height = 0.5

    if not stage.GetPrimAtPath("/World/random_terrain/start_platform"):
        start_platform = FixedCuboid(
            prim_path="/World/random_terrain/start_platform",
            position=np.array([0.0, 0.0, -platform_height / 2.0 + 0.001], dtype=np.float64),
            size=1.0,
            scale=np.array([platform_length, platform_width, platform_height], dtype=np.float64),
            color=np.array([0.15, 0.15, 0.15], dtype=np.float32),
        )
        start_platform.apply_physics_material(stair_material)

    if not stage.GetPrimAtPath("/World/random_terrain/outside_area"):
        FixedCuboid(
            prim_path="/World/random_terrain/outside_area",
            position=np.array([0.0, 0.0, -1.0], dtype=np.float64),
            size=1.0,
            scale=np.array([200.0, 200.0, 0.1], dtype=np.float64),
            color=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )

    if (not cfg_eval.random_uneven_terrain) and (not cfg_eval.random_slope_terrain):
        return

    noise_min = float(max(0.005, cfg_eval.uneven_patch_height_min))
    noise_max = float(max(noise_min + 0.003, cfg_eval.uneven_patch_height_max))
    obstacle_count = int(max(4, cfg_eval.uneven_patches_per_gap_max + 2))
    slope_min_rad = float(np.deg2rad(max(0.5, cfg_eval.slope_angle_deg_min)))
    slope_max_rad = float(np.deg2rad(max(cfg_eval.slope_angle_deg_max, cfg_eval.slope_angle_deg_min + 0.5)))

    sub_terrains: Dict[str, object] = {}
    if cfg_eval.random_uneven_terrain:
        sub_terrains["random_uniform"] = HfRandomUniformTerrainCfg(
            proportion=0.20,
            noise_range=(noise_min, noise_max),
            noise_step=max(0.0025, 0.5 * noise_min),
            downsampled_scale=0.10,
        )
        sub_terrains["discrete_obstacles"] = HfDiscreteObstaclesTerrainCfg(
            proportion=0.20,
            obstacle_width_range=(0.18, 0.55),
            obstacle_height_range=(noise_min, max(noise_max, 0.12)),
            num_obstacles=obstacle_count,
            platform_width=0.4,
        )
        sub_terrains["stairs_up"] = HfPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.04, 0.16),
            step_width=0.30,
            platform_width=0.4,
            inverted=False,
        )
        sub_terrains["stairs_down_inv"] = HfInvertedPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.04, 0.16),
            step_width=0.30,
            platform_width=0.4,
            inverted=True,
        )
    if cfg_eval.random_slope_terrain:
        sub_terrains["pyramid_slope"] = HfPyramidSlopedTerrainCfg(
            proportion=0.20,
            slope_range=(slope_min_rad, slope_max_rad),
            platform_width=0.8,
            inverted=False,
        )
    if not sub_terrains:
        return

    target_length_m = float(cfg_eval.procedural_terrain_length_m)
    target_width_m = float(cfg_eval.procedural_terrain_width_m)
    num_rows_x = int(np.ceil(target_length_m / 4.0))
    num_cols_y = int(np.ceil(target_width_m / 4.0))
    tile_size_x = float(target_length_m / max(num_rows_x, 1))
    tile_size_y = float(target_width_m / max(num_cols_y, 1))

    horizontal_scale = float(max(0.05, min(0.10, cfg_eval.slope_step_resolution_m)))
    tile_size_x = float(np.round(tile_size_x / horizontal_scale) * horizontal_scale)
    tile_size_y = float(np.round(tile_size_y / horizontal_scale) * horizontal_scale)

    terrain_cfg = TerrainGeneratorCfg(
        seed=int(np.random.randint(0, 2**31 - 1)) if seed is None else int(seed),
        curriculum=False,
        size=(tile_size_x, tile_size_y),
        border_width=0.0,
        border_height=0.0,
        num_rows=num_rows_x,
        num_cols=num_cols_y,
        horizontal_scale=horizontal_scale,
        vertical_scale=0.005,
        slope_threshold=0.75,
        sub_terrains=sub_terrains,
        difficulty_range=(0.2, 0.8),
        use_cache=False,
        color_scheme="none",
    )

    terrain_generator = TerrainGenerator(cfg=terrain_cfg, device="cpu")
    mesh = terrain_generator.terrain_mesh
    total_length = num_rows_x * tile_size_x
    center_x = (total_length / 2.0) + 1.0

    create_prim_from_mesh(
        prim_path="/World/random_terrain/mixed_track",
        mesh=mesh,
        translation=(center_x, 0.0, 0.0),
    )

    if not stage.GetPrimAtPath("/World/defaultLight").IsValid():
        light = UsdLux.DomeLight.Define(stage, "/World/defaultLight")
        light.CreateIntensityAttr().Set(1500.0)

    mesh_prim = stage.GetPrimAtPath("/World/random_terrain/mixed_track")
    if mesh_prim.IsValid():
        UsdPhysics.CollisionAPI.Apply(mesh_prim)
        mesh_col_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
        mesh_col_api.CreateApproximationAttr().Set("none")
        from pxr import Gf, UsdGeom, UsdShade

        UsdGeom.Mesh(mesh_prim).CreateDisplayColorAttr().Set([Gf.Vec3f(0.15, 0.15, 0.15)])
        material_prim = stage.GetPrimAtPath(stair_material_path)
        if material_prim.IsValid():
            material = UsdShade.Material(material_prim)
            UsdShade.MaterialBindingAPI.Apply(mesh_prim).Bind(
                material,
                UsdShade.Tokens.weakerThanDescendants,
                "physics",
            )

    print(
        f"[compare] Added procedural terrain: seed={int(terrain_cfg.seed)}, length_m={target_length_m:.2f}, "
        f"width_m={target_width_m:.2f}, tiles={num_rows_x}x{num_cols_y}"
    )



def _make_world(cfg_eval: EvalConfig, seed: Optional[int] = None):
    from src.dog.dogWorld import DogWorld

    world_cfg = DogWorld.Params()
    world_cfg.add_stairs = False
    world_cfg.add_ground_plane = False
    world = DogWorld(world_cfg)
    _add_procedural_terrain_to_world(world, cfg_eval, seed=seed)
    return world



def _configure_case(runner) -> None:
    def _apply_reflex_mode(trigger_mode, use_next_steps_enhancement: bool) -> None:
        cfg.simulation_params["reflex_trigger_mode"] = trigger_mode
        cfg.simulation_params["reflex_next_steps_height_enhancement"] = bool(use_next_steps_enhancement)
        try:
            esd = runner.mpc_controller.wb_interface.esd
            esd.trigger_mode = trigger_mode
            esd.activated = bool(trigger_mode is not False)
            esd.use_reflex_next_steps_height_enhancement = bool(use_next_steps_enhancement)
        except Exception:
            pass

    try:
        runner.rl_policy_enabled = True
        _apply_reflex_mode(False, False)
        runner.mpc_controller.enable_rl_step_height(True)
        runner.mpc_controller.set_step_height_residual(0.0)
    except Exception:
        pass



def _reset_episode(runner, cfg_eval: EvalConfig) -> None:
    from src.dog.dog_runner import SimulationState

    runner.auto_active_step = None

    runner.stage.world.reset()
    try:
        runner.hard_reset_robot_state(settle_steps=200)
    except Exception:
        pass

    runner.physics_step_counter = 0
    runner.last_torques = None
    runner.last_mpc_debug_info = {}

    try:
        runner.mpc_controller.reset_for_new_episode()
    except Exception:
        pass

    runner.state = SimulationState.INIT
    try:
        runner.state_machine()
    except Exception:
        pass

    runner.ref_base_lin_vel_H[:] = 0.0
    runner.ref_base_ang_yaw_dot = 0.0
    runner.pitch_delta = 0.0

    for _ in range(int(max(0, cfg_eval.post_reset_stand_steps))):
        runner.pre_physics()
        runner.step_physics()
        runner.stage.world.step(render=False)
        runner.post_physics()

    runner.state = SimulationState.ACTIVE
    try:
        runner.state_machine()
    except Exception:
        pass

    runner.ref_base_lin_vel_H[:] = np.array(
        [cfg_eval.command_forward, cfg_eval.command_lateral, 0.0], dtype=np.float64
    )
    runner.ref_base_ang_yaw_dot = float(cfg_eval.command_yaw)



def _run_episode(
    runner,
    cfg_eval: EvalConfig,
    terrain_seed: int,
    checkpoint_key: str,
    checkpoint_label: str,
) -> Tuple[EpisodeMetrics, Dict[str, np.ndarray]]:
    sim_dt = float(runner.sim_dt)
    step = 0
    course_complete_step = -1
    stagnation_terminated = False
    stagnation_step = -1

    track_errors: List[float] = []
    cmd_speeds: List[float] = []
    act_speeds: List[float] = []
    step_height_eff: List[float] = []
    step_height_residual: List[float] = []
    swing_time_s: List[float] = []
    swing_foot_z_trace: List[float] = []
    base_x_trace: List[float] = []
    base_z_trace: List[float] = []
    roll_error_rad: List[float] = []
    pitch_error_rad: List[float] = []
    yaw_error_rad: List[float] = []
    joint_torque_mean_abs: List[float] = []
    cot_trace: List[float] = []
    position_history: List[float] = []

    stagnation_window = int(cfg_eval.stagnation_check_window_steps)
    stagnation_threshold = float(cfg_eval.stagnation_distance_threshold)

    while step < cfg_eval.episode_length_steps:
        if cfg_eval.command_ramp_steps > 0:
            ramp = float(np.clip(step / float(cfg_eval.command_ramp_steps), 0.0, 1.0))
        else:
            ramp = 1.0
        runner.ref_base_lin_vel_H[:] = np.array(
            [cfg_eval.command_forward * ramp, cfg_eval.command_lateral * ramp, 0.0], dtype=np.float64
        )
        runner.ref_base_ang_yaw_dot = float(cfg_eval.command_yaw * ramp)

        runner.pre_physics()
        runner.step_physics()
        runner.stage.world.step(render=bool(step % cfg_eval.render_every_n_steps == 0))
        runner.post_physics()

        snap = runner.mpc_controller.get_rl_state_snapshot()
        if isinstance(snap, dict):
            pos = np.asarray(snap.get("position", np.zeros(3)), dtype=np.float64)
            lin = np.asarray(snap.get("linear_velocity", np.zeros(3)), dtype=np.float64)
            ref_lin_nom = np.asarray(snap.get("ref_base_lin_vel", np.zeros(3)), dtype=np.float64)

            cmd_speed_nom = float(np.linalg.norm(ref_lin_nom[0:2]))
            act_speed_nom = float(np.linalg.norm(lin[0:2]))
            track_errors.append(abs(act_speed_nom - cmd_speed_nom))
            cmd_speeds.append(cmd_speed_nom)
            act_speeds.append(act_speed_nom)

            base_x = float(pos[0])
            base_z = float(pos[2])
            base_x_trace.append(base_x)
            base_z_trace.append(base_z)

            euler = np.asarray(snap.get("base_ori_euler_xyz", np.zeros(3)), dtype=np.float64).reshape(-1)
            if euler.size >= 3:
                roll_error_rad.append(abs(float(np.arctan2(np.sin(euler[0]), np.cos(euler[0])))))
                pitch_error_rad.append(abs(float(np.arctan2(np.sin(euler[1]), np.cos(euler[1])))))
                yaw_error_rad.append(abs(float(np.arctan2(np.sin(euler[2]), np.cos(euler[2])))))

            torques = getattr(runner, "last_torques", None)
            power = 0.0
            torque_mean = 0.0
            if torques is not None:
                if hasattr(torques, "detach"):
                    torques_arr = np.asarray(torques.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
                else:
                    torques_arr = np.asarray(torques, dtype=np.float64).reshape(-1)
                if torques_arr.size > 0:
                    torque_mean = float(np.mean(np.abs(torques_arr)))
                joint_vel_raw = snap.get("joint_velocities", None)
                if joint_vel_raw is not None:
                    joint_vel_arr = np.asarray(joint_vel_raw, dtype=np.float64).reshape(-1)
                else:
                    joint_vel_arr = np.array([])
                if torques_arr.size >= 12 and joint_vel_arr.size >= 12:
                    power = float(np.sum(np.abs(torques_arr[:12] * joint_vel_arr[:12])))
            joint_torque_mean_abs.append(torque_mean)

            robot_mass = float(getattr(runner, "robot_mass", 15.0))
            if act_speed_nom > 0.01:
                cot = power / (robot_mass * 9.81 * act_speed_nom)
            else:
                cot = power / (robot_mass * 9.81 * 0.05)
            cot_trace.append(cot)

            nominal_h = float(getattr(runner.mpc_controller, "_nominal_step_height", 0.0))
            min_h = float(getattr(runner.mpc_controller, "_rl_step_height_min_m", nominal_h))
            max_h = float(getattr(runner.mpc_controller, "_rl_step_height_max_m", nominal_h))
            rl_enabled = bool(getattr(runner.mpc_controller, "_rl_step_height_enabled", False))
            if rl_enabled:
                applied_residual = float(getattr(runner.mpc_controller, "_rl_step_height_residual_m", 0.0))
                effective_h = float(np.clip(nominal_h + applied_residual, min_h, max_h))
                residual_applied = float(effective_h - nominal_h)
            else:
                effective_h = float(np.clip(nominal_h, min_h, max_h))
                residual_applied = 0.0
            step_height_eff.append(effective_h)
            step_height_residual.append(residual_applied)

            current_swing_time = float(
                snap.get(
                    "adaptive_swing_time",
                    getattr(
                        runner.mpc_controller.wb_interface,
                        "adaptive_swing_time",
                        cfg.simulation_params.get("adaptive_swing_time", 0.20),
                    ),
                )
            )
            swing_time_s.append(current_swing_time)

            feet_pos_w = np.asarray(snap.get("feet_pos_w", np.zeros(12)), dtype=np.float64).reshape(-1)
            current_contact = np.asarray(snap.get("current_contact", np.ones(4)), dtype=bool).reshape(-1)
            if feet_pos_w.size >= 12 and current_contact.size >= 4:
                feet_z = feet_pos_w[2::3][:4]
                swing_z = feet_z[~current_contact[:4]]
                swing_foot_z_trace.append(float(np.max(swing_z)) if swing_z.size > 0 else float("nan"))
            else:
                swing_foot_z_trace.append(float("nan"))

            position_history.append(base_x)
            if len(position_history) > stagnation_window:
                position_history.pop(0)

            if base_x >= float(cfg_eval.procedural_narrow_success_x):
                if course_complete_step < 0:
                    course_complete_step = step
                break

            stagnation_check_min_step = int(max(cfg_eval.command_ramp_steps, cfg_eval.stagnation_check_window_steps))
            in_stagnation_region = bool(base_x >= -0.2)
            if len(position_history) >= stagnation_window and step >= stagnation_check_min_step and in_stagnation_region:
                displacement = position_history[-1] - position_history[0]
                if displacement < stagnation_threshold:
                    stagnation_terminated = True
                    stagnation_step = int(step)
                    break

        step += 1

    course_complete = float(1.0 if course_complete_step >= 0 else 0.0)
    completion_time_s = float("nan")
    if course_complete_step >= 0:
        completion_time_s = float((course_complete_step + 1) * sim_dt)

    metrics = EpisodeMetrics(
        terrain="procedural",
        terrain_seed=int(terrain_seed),
        case=str(checkpoint_key),
        checkpoint_label=str(checkpoint_label),
        episode_idx=-1,
        command_forward=float(cfg_eval.command_forward),
        course_success=course_complete,
        course_complete=course_complete,
        goal_crossed=course_complete,
        completion_time_s=completion_time_s,
        strict_completion_time_s=completion_time_s,
        mean_tracking_error=float(np.mean(track_errors)) if track_errors else float("nan"),
        mean_cmd_speed_nominal=float(np.mean(cmd_speeds)) if cmd_speeds else float("nan"),
        mean_actual_speed_nominal=float(np.mean(act_speeds)) if act_speeds else float("nan"),
        mean_swing_time_s=float(np.mean(swing_time_s)) if swing_time_s else float("nan"),
        mean_cot=float(np.mean(cot_trace)) if cot_trace else float("nan"),
        mean_step_height_residual=float(np.mean(step_height_residual)) if step_height_residual else float("nan"),
        mean_base_x=float(np.mean(base_x_trace)) if base_x_trace else float("nan"),
        mean_base_z=float(np.mean(base_z_trace)) if base_z_trace else float("nan"),
        mean_swing_foot_z=float(np.nanmean(swing_foot_z_trace)) if swing_foot_z_trace else float("nan"),
        mean_joint_torque_abs=float(np.mean(joint_torque_mean_abs)) if joint_torque_mean_abs else float("nan"),
        mean_roll_error_rad=float(np.mean(roll_error_rad)) if roll_error_rad else float("nan"),
        mean_pitch_error_rad=float(np.mean(pitch_error_rad)) if pitch_error_rad else float("nan"),
        mean_yaw_error_rad=float(np.mean(yaw_error_rad)) if yaw_error_rad else float("nan"),
        total_steps=int(step + 1),
    )

    if stagnation_terminated:
        print(
            f"[compare] Episode terminated by stagnation at step={stagnation_step} "
            f"(window={int(stagnation_window)}, threshold_m={float(stagnation_threshold):.3f})"
        )

    traces = {
        "tracking_error": np.asarray(track_errors, dtype=np.float32),
        "cmd_speed": np.asarray(cmd_speeds, dtype=np.float32),
        "actual_speed": np.asarray(act_speeds, dtype=np.float32),
        "step_height_eff": np.asarray(step_height_eff, dtype=np.float32),
        "step_height_residual": np.asarray(step_height_residual, dtype=np.float32),
        "swing_time_s": np.asarray(swing_time_s, dtype=np.float32),
        "swing_foot_z": np.asarray(swing_foot_z_trace, dtype=np.float32),
        "base_x": np.asarray(base_x_trace, dtype=np.float32),
        "base_z": np.asarray(base_z_trace, dtype=np.float32),
        "roll_error_rad": np.asarray(roll_error_rad, dtype=np.float32),
        "pitch_error_rad": np.asarray(pitch_error_rad, dtype=np.float32),
        "yaw_error_rad": np.asarray(yaw_error_rad, dtype=np.float32),
        "joint_torque_mean_abs": np.asarray(joint_torque_mean_abs, dtype=np.float32),
        "cot": np.asarray(cot_trace, dtype=np.float32),
    }
    return metrics, traces



def _aggregate_case_trace(traces: List[np.ndarray]) -> np.ndarray:
    if not traces:
        return np.zeros(0, dtype=np.float32)
    max_len = max(int(trace.shape[0]) for trace in traces)
    if max_len <= 0:
        return np.zeros(0, dtype=np.float32)
    arr = np.full((len(traces), max_len), np.nan, dtype=np.float32)
    for i, trace in enumerate(traces):
        n = int(trace.shape[0])
        if n > 0:
            arr[i, :n] = trace
    return np.asarray(np.nanmean(arr, axis=0), dtype=np.float32)



def _plot_results(
    out_png: str,
    rows: List[EpisodeMetrics],
    traces_by_case: Dict[str, List[Dict[str, np.ndarray]]],
    case_specs: List[CaseSpec],
    sim_dt: float,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(11, 11), constrained_layout=True)

    episode_indices = sorted({int(r.episode_idx) for r in rows})
    if not episode_indices:
        for ax in axes:
            ax.grid(True, alpha=0.3)
        fig.savefig(out_png, dpi=140)
        return

    x = np.arange(len(episode_indices), dtype=np.float32)
    bar_w = 0.38
    for idx, spec in enumerate(case_specs):
        case_rows = [r for r in rows if r.case == spec.key]
        by_ep = {int(r.episode_idx): float(r.completion_time_s) for r in case_rows}
        vals = np.asarray([by_ep.get(ep, float("nan")) for ep in episode_indices], dtype=np.float32)
        ok = np.isfinite(vals)
        if np.any(ok):
            axes[1].bar(x[ok] + (float(idx) - (len(case_specs) - 1) / 2.0) * bar_w, vals[ok], width=bar_w, color=spec.color, alpha=0.8, label=spec.label)

    swing_time_by_case: Dict[str, np.ndarray] = {}
    swing_foot_z_by_case: Dict[str, np.ndarray] = {}
    cmd_by_case: Dict[str, np.ndarray] = {}
    act_by_case: Dict[str, np.ndarray] = {}
    max_len = 0
    for spec in case_specs:
        case_traces = traces_by_case.get(spec.key, [])
        cmd = _aggregate_case_trace([trace.get("cmd_speed", np.zeros(0, dtype=np.float32)) for trace in case_traces])
        act = _aggregate_case_trace([trace.get("actual_speed", np.zeros(0, dtype=np.float32)) for trace in case_traces])
        swing_time = _aggregate_case_trace([trace.get("swing_time_s", np.zeros(0, dtype=np.float32)) for trace in case_traces])
        swing_foot_z = _aggregate_case_trace([trace.get("swing_foot_z", np.zeros(0, dtype=np.float32)) for trace in case_traces])
        cmd_by_case[spec.key] = cmd
        act_by_case[spec.key] = act
        swing_time_by_case[spec.key] = swing_time
        swing_foot_z_by_case[spec.key] = swing_foot_z
        max_len = max(max_len, int(cmd.shape[0]), int(act.shape[0]), int(swing_time.shape[0]), int(swing_foot_z.shape[0]))

    if max_len > 0:
        t = np.arange(max_len, dtype=np.float32) * float(sim_dt)

        def _pad(a: np.ndarray, n: int) -> np.ndarray:
            if a.shape[0] >= n:
                return a[:n]
            if a.shape[0] == 0:
                return np.zeros(n, dtype=np.float32)
            return np.pad(a, (0, n - a.shape[0]), mode="edge")

        for spec in case_specs:
            cmd = _pad(cmd_by_case[spec.key], max_len)
            act = _pad(act_by_case[spec.key], max_len)
            swing_time = _pad(swing_time_by_case[spec.key], max_len)
            swing_foot_z = _pad(swing_foot_z_by_case[spec.key], max_len)
            axes[0].plot(t, cmd, "--", color=spec.color, alpha=0.55, label=f"{spec.label} commanded")
            axes[0].plot(t, act, "-", color=spec.color, linewidth=2.0, label=f"{spec.label} actual")
            axes[2].plot(t, swing_foot_z, "-", color=spec.color, linewidth=2.0, label=f"{spec.label} swing foot")
            if spec.key not in swing_time_by_case:
                swing_time = np.zeros_like(t)

        axes[0].set_title("Commanded/Actual Speed vs Time")
        axes[0].set_xlabel("Time [s]")
        axes[0].set_ylabel("Speed [m/s]")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="best")

        axes[1].set_xticks(x)
        axes[1].set_xticklabels([str(ep) for ep in episode_indices])
        axes[1].set_title("Completion-Time Histogram by Episode")
        axes[1].set_xlabel("Episode number")
        axes[1].set_ylabel("Completion time [s] (NaN = not completed)")
        axes[1].grid(True, alpha=0.3)
        if any(r.case == spec.key for spec in case_specs for r in rows):
            axes[1].legend(loc="best")

        axes[2].set_title("Swing-Foot Z Height vs Time")
        axes[2].set_xlabel("Time [s]")
        axes[2].set_ylabel("Swing foot Z [m]")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc="best")

    fig.savefig(out_png, dpi=140)



def _plot_speed_and_step_height_profiles(
    out_png: str,
    traces_by_case: Dict[str, List[Dict[str, np.ndarray]]],
    case_specs: List[CaseSpec],
    sim_dt: float,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(case_specs), 1, figsize=(12, 5.5 * len(case_specs)), constrained_layout=True)
    if len(case_specs) == 1:
        axes = np.asarray([axes])

    for ax, spec in zip(axes, case_specs):
        case_traces = traces_by_case.get(spec.key, [])
        cmd = _aggregate_case_trace([trace.get("cmd_speed", np.zeros(0, dtype=np.float32)) for trace in case_traces])
        act = _aggregate_case_trace([trace.get("actual_speed", np.zeros(0, dtype=np.float32)) for trace in case_traces])
        step_h = _aggregate_case_trace([trace.get("step_height_eff", np.zeros(0, dtype=np.float32)) for trace in case_traces])
        swing_t = _aggregate_case_trace([trace.get("swing_time_s", np.zeros(0, dtype=np.float32)) for trace in case_traces])

        max_len = max(int(cmd.shape[0]), int(act.shape[0]), int(step_h.shape[0]), int(swing_t.shape[0]))
        if max_len <= 0:
            ax.set_title(f"{spec.label}: no trace data")
            ax.grid(True, alpha=0.3)
            continue

        t = np.arange(max_len, dtype=np.float32) * float(sim_dt)

        def _pad(a: np.ndarray, n: int) -> np.ndarray:
            if a.shape[0] >= n:
                return a[:n]
            if a.shape[0] == 0:
                return np.zeros(n, dtype=np.float32)
            return np.pad(a, (0, n - a.shape[0]), mode="edge")

        cmd = _pad(cmd, max_len)
        act = _pad(act, max_len)
        step_h = _pad(step_h, max_len)
        swing_t = _pad(swing_t, max_len)

        ax.plot(t, cmd, "--", color="#2ca02c", linewidth=1.8, label="commanded speed")
        ax.plot(t, act, "-", color=spec.color, linewidth=2.0, label="actual speed")
        ax.set_ylabel("Speed [m/s]")
        ax.set_title(f"{spec.label}: Commanded/Actual Speed + Effective Step Height + Swing Time")
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        ax2.plot(t, step_h, "-", color="#9467bd", linewidth=1.5, label="effective step height")
        ax2.set_ylabel("Step height [m]")

        ax3 = ax.twinx()
        ax3.spines["right"].set_position(("outward", 48))
        ax3.plot(t, swing_t, "-", color="#ff7f0e", linewidth=1.5, label="swing time")
        ax3.set_ylabel("Swing time [s]")

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        h3, l3 = ax3.get_legend_handles_labels()
        ax.legend(h1 + h2 + h3, l1 + l2 + l3, loc="best")

    axes[-1].set_xlabel("Time [s]")
    fig.savefig(out_png, dpi=140)



def _plot_velocity_tracking_by_speed(
    out_png: str,
    command_forward: float,
    traces_by_case: Dict[str, List[Dict[str, np.ndarray]]],
    case_specs: List[CaseSpec],
    sim_dt: float,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(12, 5), constrained_layout=True)
    max_len = 0
    traces: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for spec in case_specs:
        case_traces = traces_by_case.get(spec.key, [])
        cmd = _aggregate_case_trace([trace.get("cmd_speed", np.zeros(0, dtype=np.float32)) for trace in case_traces])
        act = _aggregate_case_trace([trace.get("actual_speed", np.zeros(0, dtype=np.float32)) for trace in case_traces])
        traces[spec.key] = (cmd, act)
        max_len = max(max_len, int(cmd.shape[0]), int(act.shape[0]))

    if max_len <= 0:
        ax.set_title(f"target speed {command_forward:.2f} m/s: no trace data")
        ax.grid(True, alpha=0.3)
        fig.savefig(out_png, dpi=140)
        return

    t = np.arange(max_len, dtype=np.float32) * float(sim_dt)
    for spec in case_specs:
        cmd, act = traces.get(spec.key, (np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)))
        if cmd.shape[0] < max_len:
            cmd = np.pad(cmd, (0, max_len - cmd.shape[0]), mode="edge") if cmd.shape[0] > 0 else np.zeros(max_len, dtype=np.float32)
        if act.shape[0] < max_len:
            act = np.pad(act, (0, max_len - act.shape[0]), mode="edge") if act.shape[0] > 0 else np.zeros(max_len, dtype=np.float32)
        ax.plot(t, cmd, "--", color=spec.color, alpha=0.45, linewidth=1.5)
        ax.plot(t, act, "-", color=spec.color, linewidth=2.2, label=f"{spec.label} actual")

    ax.axhline(float(command_forward), color="#2ca02c", linestyle="--", linewidth=1.6, label="target speed")
    ax.set_title(f"target speed {command_forward:.2f} m/s: velocity tracking")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Speed [m/s]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(out_png, dpi=140)



def _plot_base_z_by_speed(
    out_png: str,
    command_forward: float,
    traces_by_case: Dict[str, List[Dict[str, np.ndarray]]],
    case_specs: List[CaseSpec],
    sim_dt: float,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(12, 5), constrained_layout=True)
    max_len = 0
    traces: Dict[str, np.ndarray] = {}
    for spec in case_specs:
        case_traces = traces_by_case.get(spec.key, [])
        base_z = _aggregate_case_trace([trace.get("base_z", np.zeros(0, dtype=np.float32)) for trace in case_traces])
        traces[spec.key] = base_z
        max_len = max(max_len, int(base_z.shape[0]))

    if max_len <= 0:
        ax.set_title(f"target speed {command_forward:.2f} m/s: base z vs time (no data)")
        ax.grid(True, alpha=0.3)
        fig.savefig(out_png, dpi=140)
        return

    t = np.arange(max_len, dtype=np.float32) * float(sim_dt)
    for spec in case_specs:
        base_z = traces.get(spec.key, np.zeros(0, dtype=np.float32))
        if base_z.shape[0] < max_len:
            base_z = np.pad(base_z, (0, max_len - base_z.shape[0]), mode="edge") if base_z.shape[0] > 0 else np.zeros(max_len, dtype=np.float32)
        ax.plot(t, base_z, "-", color=spec.color, linewidth=2.0, label=spec.label)

    ax.set_title(f"target speed {command_forward:.2f} m/s: base z vs time")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Base z [m]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(out_png, dpi=140)



def _plot_metric_by_speed(
    out_png: str,
    speed_values: List[float],
    series: Dict[str, List[float]],
    ylabel: str,
    title: str,
    x_label: str = "Target speed [m/s]",
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(9.5, 5), constrained_layout=True)
    if not speed_values:
        ax.set_title(f"{title}: no speeds")
        fig.savefig(out_png, dpi=140)
        return

    x = np.asarray(speed_values, dtype=np.float32)
    any_plotted = False
    for label, values in series.items():
        if len(values) != len(speed_values):
            continue
        y = np.asarray(values, dtype=np.float32)
        if not np.any(np.isfinite(y)):
            continue
        ax.plot(x, y, marker="o", linewidth=2.0, label=label)
        any_plotted = True

    if not any_plotted:
        ax.set_title(f"{title}: no data")
    else:
        ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(ylabel)
    if y_min is not None or y_max is not None:
        ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)
    if any_plotted:
        ax.legend(loc="best")
    fig.savefig(out_png, dpi=140)



def _plot_success_counts_by_speed(
    out_png: str,
    speed_values: List[float],
    series: Dict[str, List[int]],
    title: str,
    x_label: str = "Target speed [m/s]",
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(9.5, 5), constrained_layout=True)
    if not speed_values:
        ax.set_title(f"{title}: no speeds")
        fig.savefig(out_png, dpi=140)
        return
    if not series:
        ax.set_title(f"{title}: no data")
        fig.savefig(out_png, dpi=140)
        return

    x = np.arange(len(speed_values), dtype=np.float32)
    labels = list(series.keys())
    num_cases = max(len(labels), 1)
    width = 0.8 / float(num_cases)
    any_plotted = False
    max_count = 0

    for idx, label in enumerate(labels):
        values = series.get(label, [])
        if len(values) != len(speed_values):
            continue
        counts = np.asarray(values, dtype=np.float32)
        offset = (float(idx) - (float(num_cases) - 1.0) / 2.0) * width
        bars = ax.bar(x + offset, counts, width=width, label=label)
        any_plotted = True
        max_count = max(max_count, int(np.nanmax(counts)))
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                float(bar.get_height()) + 0.05,
                f"{int(count)}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    if not any_plotted:
        ax.set_title(f"{title}: no data")
        fig.savefig(out_png, dpi=140)
        return

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Successful episodes")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.2f}" for v in speed_values])
    ax.set_ylim(0.0, float(max(max_count, 1)) + 0.6)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(out_png, dpi=140)



def _plot_success_rate_by_speed(
    out_png: str,
    speed_values: List[float],
    series: Dict[str, List[float]],
    title: str,
    x_label: str = "Target speed [m/s]",
) -> None:
    _plot_metric_by_speed(out_png, speed_values, series, ylabel="Success rate", title=title, x_label=x_label, y_min=0.0, y_max=1.05)



def _plot_attitude_stability_by_speed(
    out_png: str,
    speed_values: List[float],
    roll_series: Dict[str, List[float]],
    pitch_series: Dict[str, List[float]],
    yaw_series: Dict[str, List[float]],
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 10.5), constrained_layout=True)
    if not speed_values:
        axes[0].set_title("Attitude stability: no speeds")
        fig.savefig(out_png, dpi=140)
        return

    def _plot_axis(ax, series, title, ylabel):
        x = np.asarray(speed_values, dtype=np.float32)
        any_plotted = False
        for label, values in series.items():
            if len(values) != len(speed_values):
                continue
            y = np.asarray(values, dtype=np.float32)
            if not np.any(np.isfinite(y)):
                continue
            ax.plot(x, y, marker="o", linewidth=2.0, label=label)
            any_plotted = True
        if not any_plotted:
            ax.set_title(f"{title}: no data")
        else:
            ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if any_plotted:
            ax.legend(loc="best")

    _plot_axis(axes[0], roll_series, "Mean roll tracking error vs speed", "Mean |roll| [rad]")
    _plot_axis(axes[1], pitch_series, "Mean pitch tracking error vs speed", "Mean |pitch| [rad]")
    _plot_axis(axes[2], yaw_series, "Mean yaw tracking error vs speed", "Mean |yaw| [rad]")
    axes[-1].set_xlabel("Target speed [m/s]")
    fig.savefig(out_png, dpi=140)



def _worker_process(
    worker_id: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    args: argparse.Namespace,
    cfg_eval: EvalConfig,
    case_specs: List[CaseSpec],
) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    import random

    import numpy as np
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    random.seed(42 + int(worker_id))
    np.random.seed(42 + int(worker_id))
    torch.manual_seed(42 + int(worker_id))

    _ensure_isaac_sim_python_path()

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": args.headless})

    import omni.usd
    # Kit startup can mutate sys.path; ensure extension roots remain visible.
    _ensure_isaac_sim_python_path()

    from src.dog.dog_runner import DogRunner

    current_seed = None
    world = None
    runner = None
    runner_checkpoint = None

    def _build_runner(checkpoint_path: str) -> DogRunner:
        return DogRunner(
            world,
            rl_residual_checkpoint=checkpoint_path,
            rl_residual_device=args.device,
            rl_residual_deterministic=not args.stochastic,
            rl_residual_warmup_steps=int(cfg.simulation_params.get("rl_residual_warmup_steps", 120)),
            rl_residual_ramp_steps=int(cfg.simulation_params.get("rl_residual_ramp_steps", 400)),
        )

    while True:
        try:
            task = task_queue.get(timeout=3)
        except queue.Empty:
            continue

        if task is None:
            break

        seed = int(task["seed"])
        ep = int(task["ep_idx"])
        command_forward = float(task["command_forward"])

        if current_seed != seed:
            if runner is not None:
                runner.shutdown()
                runner = None
                runner_checkpoint = None
            if world is not None:
                omni.usd.get_context().new_stage()
            world = _make_world(cfg_eval, seed=seed)
            current_seed = seed

        for case_spec in case_specs:
            try:
                if runner is None or runner_checkpoint != case_spec.checkpoint:
                    if runner is not None:
                        runner.shutdown()
                    runner = _build_runner(case_spec.checkpoint)
                    runner_checkpoint = case_spec.checkpoint

                cfg_eval.command_forward = float(command_forward)
                _configure_case(runner)
                _reset_episode(runner, cfg_eval)
                ep_metrics, ep_traces = _run_episode(
                    runner,
                    cfg_eval,
                    terrain_seed=seed,
                    checkpoint_key=case_spec.key,
                    checkpoint_label=case_spec.label,
                )
                ep_metrics.case = case_spec.key
                ep_metrics.checkpoint_label = case_spec.label
                ep_metrics.episode_idx = ep
                ep_metrics.command_forward = float(command_forward)

                result_queue.put(
                    {
                        "type": "result",
                        "metrics": ep_metrics,
                        "traces": ep_traces,
                        "case": case_spec.key,
                        "terrain_seed": seed,
                        "command_forward": command_forward,
                        "ep_idx": ep,
                        "sim_dt": float(runner.sim_dt),
                    }
                )
                print(
                    f"[Worker {worker_id}][procedural:{command_forward:.2f}:{case_spec.key}] ep={ep} seed={seed} "
                    f"success={ep_metrics.course_success:.0f} complete={ep_metrics.course_complete:.0f} "
                    f"time_s={ep_metrics.completion_time_s:.3f} track_err_mean={ep_metrics.mean_tracking_error:.4f} "
                    f"cot_mean={ep_metrics.mean_cot:.4f} steps={ep_metrics.total_steps}"
                )
            except Exception:
                result_queue.put(
                    {
                        "type": "error",
                        "message": f"[Worker {worker_id}][procedural:{command_forward:.2f}:{case_spec.key}] ep={ep} seed={seed}\n{traceback.format_exc()}",
                    }
                )

    try:
        if runner is not None:
            runner.shutdown()
    except Exception:
        pass
    try:
        if world is not None:
            del world
    except Exception:
        pass
    try:
        simulation_app.close()
    except Exception:
        pass



def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two MPC+RL checkpoints on randomized procedural terrain. "
            "Each episode generates one terrain seed and runs both checkpoints on that same terrain."
        )
    )
    parser.add_argument(
        "--checkpoint-a",
        type=str,
        default=str(cfg.simulation_params.get("rl_residual_checkpoint", "")),
        help="First RL residual checkpoint for comparison.",
    )
    parser.add_argument(
        "--checkpoint-b",
        type=str,
        required=False,
        default="",
        help="Second RL residual checkpoint for comparison.",
    )
    parser.add_argument(
        "--checkpoint-a-label",
        type=str,
        default="checkpoint_a",
        help="Label for the first checkpoint in plots and CSV output.",
    )
    parser.add_argument(
        "--checkpoint-b-label",
        type=str,
        default="checkpoint_b",
        help="Label for the second checkpoint in plots and CSV output.",
    )
    parser.add_argument("--episodes", type=int, default=4, help="Episodes per speed value.")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=_HEADLESS_DEFAULT)
    parser.add_argument("--num-actors", type=int, default=1, help="Number of worker processes.")
    parser.add_argument("--output-dir", type=str, default="analysis")
    parser.add_argument("--speed-min", type=float, default=None, help="Minimum forward speed for sweep.")
    parser.add_argument("--speed-max", type=float, default=None, help="Maximum forward speed for sweep.")
    parser.add_argument("--speed-step", type=float, default=None, help="Step size for speed sweep.")
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default=str(cfg.simulation_params.get("rl_residual_device", "cpu")),
        help="Device for the loaded checkpoints.",
    )
    parser.add_argument(
        "--stochastic",
        action=argparse.BooleanOptionalAction,
        default=bool(cfg.simulation_params.get("rl_residual_stochastic", False)),
        help="Use stochastic actor actions instead of deterministic means.",
    )
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    if int(args.num_actors) < 1:
        raise ValueError("--num-actors must be >= 1")

    ckpt_a = _resolve_checkpoint_path(args.checkpoint_a)
    ckpt_b = _resolve_checkpoint_path(args.checkpoint_b)
    if not ckpt_a:
        raise ValueError("No first checkpoint configured. Set --checkpoint-a or simulation_params['rl_residual_checkpoint'].")
    if not os.path.exists(ckpt_a):
        raise FileNotFoundError(f"Checkpoint A not found: {ckpt_a}")
    if not ckpt_b:
        raise ValueError("No second checkpoint configured. Set --checkpoint-b.")
    if not os.path.exists(ckpt_b):
        raise FileNotFoundError(f"Checkpoint B not found: {ckpt_b}")
    if os.path.abspath(ckpt_a) == os.path.abspath(ckpt_b):
        print("[compare] Warning: checkpoint-a and checkpoint-b resolve to the same file.")

    case_specs = [
        CaseSpec("checkpoint_a", str(args.checkpoint_a_label), "#1f77b4", ckpt_a),
        CaseSpec("checkpoint_b", str(args.checkpoint_b_label), "#d62728", ckpt_b),
    ]

    os.makedirs(args.output_dir, exist_ok=True)
    stamp = int(time.time())
    out_csv = os.path.join(args.output_dir, f"procedural_mpc_rl_checkpoint_compare_{stamp}.csv")

    cfg_eval = EvalConfig(episodes_per_speed=int(args.episodes))
    if args.speed_min is not None:
        cfg_eval.speed_min = float(args.speed_min)
    if args.speed_max is not None:
        cfg_eval.speed_max = float(args.speed_max)
    if args.speed_step is not None:
        cfg_eval.speed_step = float(args.speed_step)

    speed_values = _build_speed_sweep(cfg_eval.speed_min, cfg_eval.speed_max, cfg_eval.speed_step)

    tasks = []
    for command_forward in speed_values:
        for ep in range(cfg_eval.episodes_per_speed):
            tasks.append(
                {
                    "command_forward": float(command_forward),
                    "ep_idx": ep,
                    "seed": int(np.random.randint(0, 2**31 - 1)),
                }
            )

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()

    for task in tasks:
        task_queue.put(task)
    num_workers = min(int(args.num_actors), len(tasks)) if tasks else 0
    for _ in range(num_workers):
        task_queue.put(None)

    workers = []
    for worker_idx in range(num_workers):
        proc = ctx.Process(target=_worker_process, args=(worker_idx, task_queue, result_queue, args, cfg_eval, case_specs))
        proc.start()
        workers.append(proc)
        time.sleep(2.0)

    expected_results = len(tasks) * len(case_specs)
    results_collected = 0
    all_rows: List[EpisodeMetrics] = []
    all_results_raw = []
    case_errors: List[str] = []
    traces_by_case: Dict[str, List[Dict[str, np.ndarray]]] = {spec.key: [] for spec in case_specs}
    sim_dt_captured = 0.01

    while results_collected < expected_results:
        try:
            res = result_queue.get(timeout=1.0)
        except queue.Empty:
            if not any(proc.is_alive() for proc in workers):
                print("[compare] All workers have exited unexpectedly.")
                break
            continue

        if res["type"] == "result":
            ep_metrics = res["metrics"]
            ep_traces = res["traces"]
            case_name = str(res["case"])
            all_rows.append(ep_metrics)
            traces_by_case.setdefault(case_name, []).append(ep_traces)
            all_results_raw.append(res)
            sim_dt_captured = float(res["sim_dt"])
            results_collected += 1
        elif res["type"] == "error":
            case_errors.append(str(res["message"]))
            results_collected += 1

    for proc in workers:
        proc.join(timeout=5.0)
        if proc.is_alive():
            proc.terminate()

    if not all_rows:
        raise RuntimeError("No completed episodes were recorded.")

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "terrain",
                "terrain_seed",
                "checkpoint_key",
                "checkpoint_label",
                "episode_idx",
                "command_forward",
                "course_success",
                "course_complete",
                "goal_crossed",
                "completion_time_s",
                "strict_completion_time_s",
                "mean_tracking_error",
                "mean_cmd_speed_nominal",
                "mean_actual_speed_nominal",
                "mean_swing_time_s",
                "mean_cot",
                "mean_step_height_residual",
                "mean_base_x",
                "mean_base_z",
                "mean_swing_foot_z",
                "mean_joint_torque_abs",
                "mean_roll_error_rad",
                "mean_pitch_error_rad",
                "mean_yaw_error_rad",
                "total_steps",
            ]
        )
        for row in all_rows:
            writer.writerow(
                [
                    row.terrain,
                    row.terrain_seed,
                    row.case,
                    row.checkpoint_label,
                    row.episode_idx,
                    row.command_forward,
                    row.course_success,
                    row.course_complete,
                    row.goal_crossed,
                    row.completion_time_s,
                    row.strict_completion_time_s,
                    row.mean_tracking_error,
                    row.mean_cmd_speed_nominal,
                    row.mean_actual_speed_nominal,
                    row.mean_swing_time_s,
                    row.mean_cot,
                    row.mean_step_height_residual,
                    row.mean_base_x,
                    row.mean_base_z,
                    row.mean_swing_foot_z,
                    row.mean_joint_torque_abs,
                    row.mean_roll_error_rad,
                    row.mean_pitch_error_rad,
                    row.mean_yaw_error_rad,
                    row.total_steps,
                ]
            )

    def _row_matches(row: EpisodeMetrics, command_forward: float, case: str) -> bool:
        return row.terrain == "procedural" and row.case == case and abs(float(row.command_forward) - float(command_forward)) < 1e-9

    def _mean_metric(command_forward: float, case: str, attr: str) -> float:
        rows = [row for row in all_rows if _row_matches(row, command_forward, case)]
        values = [getattr(row, attr) for row in rows]
        return _safe_nanmean(values) if values else float("nan")

    def _mean_trace(command_forward: float, case: str, trace_key: str) -> float:
        values: List[float] = []
        for task_res in all_results_raw:
            if task_res["case"] != case:
                continue
            if abs(float(task_res.get("command_forward", command_forward)) - float(command_forward)) >= 1e-9:
                continue
            trace = task_res.get("traces", {})
            arr = np.asarray(trace.get(trace_key, np.zeros(0, dtype=np.float32)), dtype=np.float32)
            finite = arr[np.isfinite(arr)]
            if finite.size > 0:
                values.append(float(np.mean(finite)))
        return _safe_nanmean(values) if values else float("nan")

    def _mean_base_x(command_forward: float, case: str) -> float:
        values: List[float] = []
        for task_res in all_results_raw:
            if task_res["case"] != case:
                continue
            if abs(float(task_res.get("command_forward", command_forward)) - float(command_forward)) >= 1e-9:
                continue
            trace = task_res.get("traces", {})
            base_x = np.asarray(trace.get("base_x", np.zeros(0, dtype=np.float32)), dtype=np.float32)
            if base_x.size > 0:
                values.append(float(np.mean(base_x)))
        return _safe_nanmean(values) if values else float("nan")

    print("\n[compare] Summary")
    for command_forward in speed_values:
        print(f"  Command forward={command_forward:.3f} m/s")
        for spec in case_specs:
            rows = [row for row in all_rows if _row_matches(row, command_forward, spec.key)]
            success_rate = float(np.mean([row.course_success for row in rows])) if rows else float("nan")
            complete_rate = float(np.mean([row.course_complete for row in rows])) if rows else float("nan")
            mean_time = _safe_nanmean([row.completion_time_s for row in rows]) if rows else float("nan")
            mean_track = _safe_nanmean([row.mean_tracking_error for row in rows]) if rows else float("nan")
            mean_cot = _safe_nanmean([row.mean_cot for row in rows]) if rows else float("nan")
            print(
                f"    {spec.label:<16} success_rate={success_rate:.3f}, complete_rate={complete_rate:.3f}, "
                f"completion_time_s={mean_time:.3f}, mean_tracking_error={mean_track:.4f}, mean_cot={mean_cot:.4f}"
            )

        speed_rows = [row for row in all_rows if abs(float(row.command_forward) - float(command_forward)) < 1e-9]
        if not speed_rows:
            continue

        traces_for_speed: Dict[str, List[Dict[str, np.ndarray]]] = {spec.key: [] for spec in case_specs}
        for task_res in all_results_raw:
            if abs(float(task_res.get("command_forward", command_forward)) - float(command_forward)) < 1e-9:
                traces_for_speed[str(task_res["case"])].append(task_res["traces"])

        out_png_tracking = os.path.join(args.output_dir, f"procedural_mpc_rl_ckpt_compare_speed_{command_forward:.2f}_{stamp}_tracking.png")
        out_png_base_z = os.path.join(args.output_dir, f"procedural_mpc_rl_ckpt_compare_speed_{command_forward:.2f}_{stamp}_base_z.png")
        out_png_overview = os.path.join(args.output_dir, f"procedural_mpc_rl_ckpt_compare_speed_{command_forward:.2f}_{stamp}_overview.png")
        out_png_profiles = os.path.join(args.output_dir, f"procedural_mpc_rl_ckpt_compare_speed_{command_forward:.2f}_{stamp}_profiles.png")

        _plot_velocity_tracking_by_speed(out_png_tracking, command_forward, traces_for_speed, case_specs, sim_dt=float(sim_dt_captured))
        _plot_base_z_by_speed(out_png_base_z, command_forward, traces_for_speed, case_specs, sim_dt=float(sim_dt_captured))
        _plot_results(out_png_overview, speed_rows, traces_for_speed, case_specs, sim_dt=float(sim_dt_captured))
        _plot_speed_and_step_height_profiles(out_png_profiles, traces_for_speed, case_specs, sim_dt=float(sim_dt_captured))

    case_labels = {spec.key: spec.label for spec in case_specs}
    tracking_series: Dict[str, List[float]] = {}
    success_series: Dict[str, List[float]] = {}
    completion_series: Dict[str, List[float]] = {}
    residual_series: Dict[str, List[float]] = {}
    swing_series: Dict[str, List[float]] = {}
    base_x_series: Dict[str, List[float]] = {}
    base_z_series: Dict[str, List[float]] = {}
    swing_foot_z_series: Dict[str, List[float]] = {}
    roll_series: Dict[str, List[float]] = {}
    pitch_series: Dict[str, List[float]] = {}
    yaw_series: Dict[str, List[float]] = {}
    torque_series: Dict[str, List[float]] = {}
    cot_series: Dict[str, List[float]] = {}

    for case_key, label in case_labels.items():
        tracking_series[label] = [_mean_metric(speed, case_key, "mean_tracking_error") for speed in speed_values]
        success_series[label] = [_mean_metric(speed, case_key, "course_success") for speed in speed_values]
        completion_series[label] = [_mean_metric(speed, case_key, "completion_time_s") for speed in speed_values]
        residual_series[label] = [_mean_metric(speed, case_key, "mean_step_height_residual") for speed in speed_values]
        roll_series[label] = [_mean_trace(speed, case_key, "roll_error_rad") for speed in speed_values]
        pitch_series[label] = [_mean_trace(speed, case_key, "pitch_error_rad") for speed in speed_values]
        yaw_series[label] = [_mean_trace(speed, case_key, "yaw_error_rad") for speed in speed_values]
        torque_series[label] = [_mean_trace(speed, case_key, "joint_torque_mean_abs") for speed in speed_values]
        base_z_series[label] = [_mean_trace(speed, case_key, "base_z") for speed in speed_values]
        cot_series[label] = [_mean_metric(speed, case_key, "mean_cot") for speed in speed_values]
        swing_foot_z_series[label] = [_mean_trace(speed, case_key, "swing_foot_z") for speed in speed_values]
        swing_series[label] = [_mean_metric(speed, case_key, "mean_swing_time_s") for speed in speed_values]
        base_x_series[label] = [_mean_base_x(speed, case_key) for speed in speed_values]

    out_tracking = os.path.join(args.output_dir, f"procedural_mpc_rl_ckpt_tracking_error_vs_speed_{stamp}.png")
    out_completion = os.path.join(args.output_dir, f"procedural_mpc_rl_ckpt_completion_time_vs_speed_{stamp}.png")
    out_success_counts = os.path.join(args.output_dir, f"procedural_mpc_rl_ckpt_success_counts_vs_speed_{stamp}.png")
    out_success_rate = os.path.join(args.output_dir, f"procedural_mpc_rl_ckpt_success_rate_vs_speed_{stamp}.png")
    out_residual = os.path.join(args.output_dir, f"procedural_mpc_rl_ckpt_step_height_residual_vs_speed_{stamp}.png")
    out_attitude = os.path.join(args.output_dir, f"procedural_mpc_rl_ckpt_attitude_error_vs_speed_{stamp}.png")
    out_torque = os.path.join(args.output_dir, f"procedural_mpc_rl_ckpt_mean_joint_torque_vs_speed_{stamp}.png")
    out_cot = os.path.join(args.output_dir, f"procedural_mpc_rl_ckpt_cot_vs_speed_{stamp}.png")
    out_base_z_speed = os.path.join(args.output_dir, f"procedural_mpc_rl_ckpt_base_z_vs_speed_{stamp}.png")
    out_swing_foot_z_speed = os.path.join(args.output_dir, f"procedural_mpc_rl_ckpt_swing_foot_z_vs_speed_{stamp}.png")
    out_base_x = os.path.join(args.output_dir, f"procedural_mpc_rl_ckpt_base_x_vs_speed_{stamp}.png")
    out_swing = os.path.join(args.output_dir, f"procedural_mpc_rl_ckpt_swing_time_vs_speed_{stamp}.png")

    _plot_metric_by_speed(out_tracking, speed_values, tracking_series, ylabel="Mean tracking error [m/s]", title="Mean speed tracking error vs speed")
    _plot_metric_by_speed(out_completion, speed_values, completion_series, ylabel="Mean completion time [s]", title="Mean completion time vs speed")
    success_counts_series: Dict[str, List[int]] = {}
    for case_key, label in case_labels.items():
        counts: List[int] = []
        for speed in speed_values:
            rows = [
                row
                for row in all_rows
                if row.case == case_key and abs(float(row.command_forward) - float(speed)) < 1e-9
            ]
            counts.append(int(sum(1 for row in rows if float(row.course_success) > 0.5)))
        success_counts_series[label] = counts

    _plot_success_counts_by_speed(
        out_success_counts,
        speed_values,
        success_counts_series,
        title="Successful episodes by speed and checkpoint",
    )
    _plot_success_rate_by_speed(out_success_rate, speed_values, success_series, title="Success rate vs speed")
    _plot_metric_by_speed(out_residual, speed_values, residual_series, ylabel="Mean step height residual [m]", title="Mean step height residual vs speed")
    _plot_attitude_stability_by_speed(out_attitude, speed_values, roll_series, pitch_series, yaw_series)
    _plot_metric_by_speed(out_torque, speed_values, torque_series, ylabel="Mean joint torque |tau| [Nm]", title="Mean joint torque magnitude vs speed")
    _plot_metric_by_speed(out_cot, speed_values, cot_series, ylabel="Cost of Transport", title="Cost of Transport (CoT) vs speed")
    _plot_metric_by_speed(out_base_z_speed, speed_values, base_z_series, ylabel="Mean base z [m]", title="Mean base z vs speed")
    _plot_metric_by_speed(out_swing_foot_z_speed, speed_values, swing_foot_z_series, ylabel="Mean swing foot z [m]", title="Mean swing foot z vs speed")
    _plot_metric_by_speed(out_base_x, speed_values, base_x_series, ylabel="Mean base x reached [m]", title="Mean base x reached vs speed")
    _plot_metric_by_speed(out_swing, speed_values, swing_series, ylabel="Mean swing time [s]", title="Mean swing time vs speed")

    recorded_cases = {row.case for row in all_rows}
    if set(case_labels.keys()).issubset(recorded_cases):
        out_png = os.path.join(args.output_dir, f"procedural_mpc_rl_ckpt_plots_{stamp}.png")
        out_png_profiles = os.path.join(args.output_dir, f"procedural_mpc_rl_ckpt_speed_stepheight_{stamp}.png")
        _plot_results(out_png, all_rows, traces_by_case=traces_by_case, case_specs=case_specs, sim_dt=float(sim_dt_captured))
        _plot_speed_and_step_height_profiles(out_png_profiles, traces_by_case=traces_by_case, case_specs=case_specs, sim_dt=float(sim_dt_captured))

    print(f"[compare] Saved CSV : {out_csv}")
    print("[compare] Plots saved in output-dir" if set(case_labels.keys()).issubset(recorded_cases) else "[compare] Plot skipped: both checkpoints were not completed for at least one speed.")

    if case_errors:
        print("[compare] Case errors:")
        for message in case_errors:
            print(f"  - {message}")


if __name__ == "__main__":
    main()

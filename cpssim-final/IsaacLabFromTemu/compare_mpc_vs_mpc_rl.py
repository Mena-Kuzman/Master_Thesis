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
from dataclasses import dataclass, replace
from typing import Dict, List, Tuple

import numpy as np


# Match run_simulation.sh runtime library fix so direct python invocation works.
_REQUIRED_CASADI_LIBDIR = "/home/mena/miniconda3/envs/pympc/lib"
_ld = os.environ.get("LD_LIBRARY_PATH", "")
_ld_parts = [p for p in _ld.split(":") if p]
if _REQUIRED_CASADI_LIBDIR not in _ld_parts:
    new_env = dict(os.environ)
    new_ld = ":".join([_REQUIRED_CASADI_LIBDIR] + _ld_parts)
    new_env["LD_LIBRARY_PATH"] = new_ld
    # Re-exec so dynamic linker picks up the new lib path before importing Isaac/CasADi.
    os.execvpe(sys.executable, [sys.executable] + sys.argv, new_env)


# Parse --headless before SimulationApp creation (same pattern as go2_pympc.py)
_HEADLESS_DEFAULT = True
if "--no-headless" in sys.argv:
    _HEADLESS_DEFAULT = False
elif "--headless" in sys.argv:
    _HEADLESS_DEFAULT = True


# Workspace imports
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if os.path.join(REPO_ROOT, "pympc") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "pympc"))
if os.path.dirname(__file__) not in sys.path:
    sys.path.insert(0, os.path.dirname(__file__))

from quadruped_pympc import config as cfg
from src.rl_config import RLTrainArgs


@dataclass
class EvalConfig:
    episodes_per_case: int = 10
    episode_length_steps: int = 20000
    render_every_n_steps: int = 8

    # Match training's fixed-course setup.
    stair_start_x: float = 1.5
    stair_step_height: float = 0.14
    stair_step_height_min: float = 0.09
    stair_step_height_max: float = 0.15
    stair_step_height_step: float = 0.02
    random_stair_step_height_min: float = 0.09
    random_stair_step_height_max: float = 0.15
    stair_step_width: float = 0.40
    stair_num_steps_up: int = 3
    stair_num_steps_down: int = 3
    stair_num_sets: int = 2
    stair_gap_distance: float = 2.0

    # Match training's command defaults.
    command_forward: float = 0.8
    command_lateral: float = 0.0
    command_yaw: float = 0.0
    speed_min: float = 0.5
    speed_max: float = 1.0
    speed_step: float = 0.1

    # Match training course-complete logic.
    descend_down_end_x_margin_m_scale: float = 0.75
    procedural_narrow_success_x: float = float(RLTrainArgs.procedural_narrow_success_x)
    survival_success_duration_steps: int = int(RLTrainArgs.survival_success_duration_steps)
    success_reset_hold_steps: int = 20
    success_reset_min_episode_step: int = 120
    success_reset_x_margin_m: float = 0.10

    # Procedural terrain settings (aligned with training defaults).
    procedural_terrain_length_m: float = float(RLTrainArgs.procedural_terrain_length_m)
    procedural_terrain_width_m: float = float(RLTrainArgs.procedural_terrain_width_m)
    random_uneven_terrain: bool = bool(RLTrainArgs.random_uneven_terrain)
    uneven_patch_height_min: float = float(RLTrainArgs.uneven_patch_height_min)
    uneven_patch_height_max: float = float(RLTrainArgs.uneven_patch_height_max)
    uneven_patches_per_gap_max: int = int(RLTrainArgs.uneven_patches_per_gap_max)
    random_slope_terrain: bool = bool(RLTrainArgs.random_slope_terrain)
    slope_angle_deg_min: float = float(RLTrainArgs.slope_angle_deg_min)
    slope_angle_deg_max: float = float(RLTrainArgs.slope_angle_deg_max)
    slope_step_resolution_m: float = float(RLTrainArgs.slope_step_resolution_m)

    # Stagnation detection for early termination.
    stagnation_check_window_steps: int = 500
    stagnation_distance_threshold: float = 0.05

    # Reset/settle tuning to mimic training env behavior.
    post_reset_stand_steps: int = 15
    command_ramp_steps: int = 40

    # Optional unmodeled payload (outside articulation, fixed-joint to base).
    payload_mass_kg: float = 7.0
    payload_size_xyz_m: Tuple[float, float, float] = (0.20, 0.12, 0.08)
    payload_offset_xyz_m: Tuple[float, float, float] = (0.0, 0.0, 0.12)



@dataclass
class EpisodeMetrics:
    terrain: str
    command_forward: float
    stair_step_height: float
    case: str
    episode_idx: int
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
    total_steps: int


def _random_stair_nominal_height(cfg_eval: EvalConfig) -> float:
    min_h = float(cfg_eval.random_stair_step_height_min)
    max_h = float(cfg_eval.random_stair_step_height_max)
    if max_h < min_h:
        min_h, max_h = max_h, min_h
    return 0.5 * (min_h + max_h)


def _sample_random_stair_heights(
    rng: np.random.Generator,
    num_steps: int,
    min_h: float,
    max_h: float,
) -> List[float]:
    if num_steps <= 0:
        return []
    heights = rng.uniform(min_h, max_h, size=int(num_steps)).astype(np.float64)
    return [float(h) for h in heights]


def _add_stair_set_to_world(
    world: DogWorld,
    set_idx: int,
    start_x: float,
    step_height: float,
    step_width: float,
    step_depth: float,
    num_steps_up: int,
    num_steps_down: int,
) -> None:
    from isaacsim.core.api.materials import PhysicsMaterial
    from isaacsim.core.api.objects import FixedCuboid
    from pxr import PhysxSchema

    stair_material_path = "/World/Physics_Materials/StairMaterial"
    stair_material = PhysicsMaterial(
        stair_material_path,
        static_friction=0.5,
        dynamic_friction=0.5,
        restitution=0.0,
    )
    stair_material_prim = world.stage.GetPrimAtPath(stair_material_path)
    PhysxSchema.PhysxMaterialAPI.Apply(stair_material_prim).CreateFrictionCombineModeAttr().Set("max")

    for i in range(num_steps_up):
        x_pos = float(start_x + i * step_width)
        z_pos = float(step_height / 2 + i * step_height)
        step = FixedCuboid(
            prim_path=f"/World/stairs_set_{set_idx}_up_step_{i}",
            position=np.array([x_pos, 0.0, z_pos], dtype=np.float64),
            size=1.0,
            scale=np.array([float(step_width), float(step_depth), float(step_height)], dtype=np.float64),
            color=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )
        step.apply_physics_material(stair_material)

    down_start_x = float(start_x + num_steps_up * step_width)
    initial_height = float(step_height * num_steps_up)
    for i in range(num_steps_down):
        x_pos = float(down_start_x + i * step_width)
        z_pos = float(initial_height - (step_height / 2 + i * step_height))
        step = FixedCuboid(
            prim_path=f"/World/stairs_set_{set_idx}_down_step_{i}",
            position=np.array([x_pos, 0.0, z_pos], dtype=np.float64),
            size=1.0,
            scale=np.array([float(step_width), float(step_depth), float(step_height)], dtype=np.float64),
            color=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )
        step.apply_physics_material(stair_material)


def _add_random_stair_set_to_world(
    world: DogWorld,
    set_idx: int,
    start_x: float,
    step_width: float,
    step_depth: float,
    step_heights_up: List[float],
    step_heights_down: List[float],
) -> None:
    from isaacsim.core.api.materials import PhysicsMaterial
    from isaacsim.core.api.objects import FixedCuboid
    from pxr import PhysxSchema

    stair_material_path = "/World/Physics_Materials/StairMaterial"
    stair_material = PhysicsMaterial(
        stair_material_path,
        static_friction=0.5,
        dynamic_friction=0.5,
        restitution=0.0,
    )
    stair_material_prim = world.stage.GetPrimAtPath(stair_material_path)
    PhysxSchema.PhysxMaterialAPI.Apply(stair_material_prim).CreateFrictionCombineModeAttr().Set("max")

    cumulative_height = 0.0
    for i, step_height in enumerate(step_heights_up):
        step_height = float(step_height)
        cumulative_height += step_height
        x_pos = float(start_x + i * step_width)
        z_pos = float(cumulative_height - 0.5 * step_height)
        step = FixedCuboid(
            prim_path=f"/World/random_stairs_set_{set_idx}_up_step_{i}",
            position=np.array([x_pos, 0.0, z_pos], dtype=np.float64),
            size=1.0,
            scale=np.array([float(step_width), float(step_depth), float(step_height)], dtype=np.float64),
            color=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )
        step.apply_physics_material(stair_material)

    down_start_x = float(start_x + len(step_heights_up) * step_width)
    current_height = float(cumulative_height)
    for i, step_height in enumerate(step_heights_down):
        step_height = float(step_height)
        x_pos = float(down_start_x + i * step_width)
        z_pos = float(current_height - 0.5 * step_height)
        step = FixedCuboid(
            prim_path=f"/World/random_stairs_set_{set_idx}_down_step_{i}",
            position=np.array([x_pos, 0.0, z_pos], dtype=np.float64),
            size=1.0,
            scale=np.array([float(step_width), float(step_depth), float(step_height)], dtype=np.float64),
            color=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )
        step.apply_physics_material(stair_material)
        current_height -= step_height


def _remove_prim_if_present(world: DogWorld, prim_path: str) -> None:
    prim = world.stage.GetPrimAtPath(prim_path)
    if prim and prim.IsValid():
        world.stage.RemovePrim(prim_path)


def _add_random_stairs_to_world(
    world: DogWorld,
    cfg_eval: EvalConfig,
    step_depth: float,
    seed: int | None = None,
) -> None:
    rng = np.random.default_rng(seed)
    num_sets = int(max(1, cfg_eval.stair_num_sets))
    num_steps_up = int(cfg_eval.stair_num_steps_up)
    num_steps_down = int(cfg_eval.stair_num_steps_down)
    min_h = float(cfg_eval.random_stair_step_height_min)
    max_h = float(cfg_eval.random_stair_step_height_max)
    if max_h < min_h:
        min_h, max_h = max_h, min_h

    for set_idx in range(num_sets):
        for i in range(num_steps_up):
            _remove_prim_if_present(world, f"/World/random_stairs_set_{set_idx}_up_step_{i}")
        for i in range(num_steps_down):
            _remove_prim_if_present(world, f"/World/random_stairs_set_{set_idx}_down_step_{i}")

    current_x = float(cfg_eval.stair_start_x)
    single_set_length = float((num_steps_up + num_steps_down) * cfg_eval.stair_step_width)

    for set_idx in range(num_sets):
        heights_up = _sample_random_stair_heights(rng, num_steps_up, min_h, max_h)
        if num_steps_down == num_steps_up and heights_up:
            heights_down = list(reversed(heights_up))
        else:
            heights_down = _sample_random_stair_heights(rng, num_steps_down, min_h, max_h)
            total_up = float(np.sum(heights_up)) if heights_up else 0.0
            total_down = float(np.sum(heights_down)) if heights_down else 0.0
            if total_down > 1e-9:
                scale = total_up / total_down if total_up > 0.0 else 1.0
                heights_down = [float(h * scale) for h in heights_down]
            heights_down = [float(np.clip(h, min_h, max_h)) for h in heights_down]
            if heights_down and total_up > 0.0:
                diff = total_up - sum(heights_down)
                heights_down[-1] = float(np.clip(heights_down[-1] + diff, min_h, max_h))

        _add_random_stair_set_to_world(
            world=world,
            set_idx=set_idx,
            start_x=current_x,
            step_width=float(cfg_eval.stair_step_width),
            step_depth=float(step_depth),
            step_heights_up=heights_up,
            step_heights_down=heights_down,
        )
        current_x += single_set_length + float(cfg_eval.stair_gap_distance)

    print(
        f"[compare] Added random stairs: total_sets={num_sets}, "
        f"height_range=[{min_h:.3f}, {max_h:.3f}]"
    )


def _add_procedural_terrain_to_world(world: DogWorld, cfg_eval: EvalConfig, seed=None) -> None:
    from isaacsim.core.api.materials import PhysicsMaterial
    from isaacsim.core.api.objects import FixedCuboid
    from pxr import PhysxSchema, UsdPhysics, UsdLux

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
    # Only remove the ground plane if it is not managed by the World scene.
    if not bool(getattr(getattr(world, "params", None), "add_ground_plane", False)):
        _remove_prim_if_present(world, "/World/defaultGroundPlane")

    stage = world.stage

    # Create Physics Material for starting platform and ensure the random_terrain Xform exists
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

    # Ensure the random_terrain group exists so the start platform is always created
    if not stage.GetPrimAtPath("/World/random_terrain"):
        stage.DefinePrim("/World/random_terrain", "Xform")

    # Starting Platform (center spawn area) - always created so robot has a stable spawn
    platform_length = 2.0
    platform_width = 2.0
    platform_height = 0.5

    # Only recreate the start platform if it doesn't already exist
    if not stage.GetPrimAtPath("/World/random_terrain/start_platform"):
        start_platform = FixedCuboid(
            prim_path="/World/random_terrain/start_platform",
            position=np.array([0.0, 0.0, -platform_height / 2.0 + 0.001], dtype=np.float64),
            size=1.0,
            scale=np.array([platform_length, platform_width, platform_height], dtype=np.float64),
            color=np.array([0.15, 0.15, 0.15], dtype=np.float32),
        )
        start_platform.apply_physics_material(stair_material)

    # Add a large plane underneath to serve as the outside area background
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
    # Snap tile sizes to be exact multiples of horizontal_scale to prevent gaps between tiles
    tile_size_x = float(np.round(tile_size_x / horizontal_scale) * horizontal_scale)
    tile_size_y = float(np.round(tile_size_y / horizontal_scale) * horizontal_scale)

    terrain_cfg = TerrainGeneratorCfg(
        seed=int(np.random.randint(0, 2**31 - 1)) if seed is None else seed,
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
        f"[compare] Added procedural terrain: length_m={target_length_m:.2f}, width_m={target_width_m:.2f}, "
        f"tiles={num_rows_x}x{num_cols_y}"
    )


def _add_unmodeled_payload_to_world(world: DogWorld, cfg_eval: EvalConfig) -> None:
    payload_mass = float(getattr(cfg_eval, "payload_mass_kg", 0.0))
    if payload_mass <= 0.0:
        return

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, PhysxSchema

    stage = world.stage
    base_path = "/World/Go2/base"
    payload_root = "/World/Payloads"
    payload_path = f"{payload_root}/Go2_payload"
    joint_path = f"{payload_root}/Go2_payload_joint"

    base_prim = stage.GetPrimAtPath(base_path)
    if not base_prim or not base_prim.IsValid():
        print(f"[compare] Payload skipped: base prim not found at {base_path}")
        return

    if not stage.GetPrimAtPath(payload_root).IsValid():
        stage.DefinePrim(payload_root, "Xform")

    # Remove stale payload/joint if they already exist.
    for path in (joint_path, payload_path):
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            stage.RemovePrim(path)

    # Compute base pose to seed payload pose close to the fixed joint target.
    base_xform = UsdGeom.Xformable(base_prim)
    base_mat = base_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    base_pos = base_mat.ExtractTranslation()
    base_rot = base_mat.ExtractRotation()
    base_quat = base_rot.GetQuat()

    offset_local = Gf.Vec3d(*cfg_eval.payload_offset_xyz_m)
    offset_world = base_rot.TransformDir(offset_local)
    payload_pos = base_pos + offset_world

    # Create payload geometry (dynamic rigid body).
    payload_geom = UsdGeom.Cube.Define(stage, payload_path)
    payload_geom.CreateSizeAttr(1.0)
    payload_xform = UsdGeom.Xformable(payload_geom)
    payload_xform.AddTranslateOp().Set(Gf.Vec3d(payload_pos))
    payload_xform.AddOrientOp().Set(Gf.Quatf(base_quat))
    payload_xform.AddScaleOp().Set(Gf.Vec3f(*cfg_eval.payload_size_xyz_m))

    payload_prim = payload_geom.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(payload_prim)
    PhysxSchema.PhysxRigidBodyAPI.Apply(payload_prim)
    UsdPhysics.CollisionAPI.Apply(payload_prim)

    mass_api = UsdPhysics.MassAPI.Apply(payload_prim)
    mass_api.CreateMassAttr().Set(payload_mass)
    mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    size_x, size_y, size_z = [float(v) for v in cfg_eval.payload_size_xyz_m]
    ixx = payload_mass * (size_y * size_y + size_z * size_z) / 12.0
    iyy = payload_mass * (size_x * size_x + size_z * size_z) / 12.0
    izz = payload_mass * (size_x * size_x + size_y * size_y) / 12.0
    mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(ixx, iyy, izz))

    # Make the payload visible (dark red).
    UsdGeom.Gprim(payload_prim).CreateDisplayColorAttr().Set([Gf.Vec3f(0.55, 0.05, 0.05)])

    # Fixed joint: base -> payload (offset in base frame, payload at its center).
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(base_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(payload_path)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*cfg_eval.payload_offset_xyz_m))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    physx_joint = PhysxSchema.PhysxJointAPI.Apply(joint.GetPrim())
    if hasattr(physx_joint, "CreateEnableCollisionAttr"):
        physx_joint.CreateEnableCollisionAttr().Set(False)
    elif hasattr(physx_joint, "CreateCollisionEnabledAttr"):
        physx_joint.CreateCollisionEnabledAttr().Set(False)

    print(
        "[compare] Added unmodeled payload: "
        f"mass_kg={payload_mass:.2f}, size_m={cfg_eval.payload_size_xyz_m}, "
        f"offset_m={cfg_eval.payload_offset_xyz_m}"
    )


def _make_world(cfg_eval: EvalConfig, terrain: str, seed: int | None = None) -> DogWorld:
    from src.dog.dogWorld import DogWorld

    world_cfg = DogWorld.Params()
    is_stairs = terrain == "stairs"
    is_random_stairs = terrain == "random_stairs"
    world_cfg.add_stairs = bool(is_stairs)
    world_cfg.add_ground_plane = bool(is_stairs or is_random_stairs)
    world_cfg.stair_start_x = float(cfg_eval.stair_start_x)
    world_cfg.stair_step_height = (
        float(_random_stair_nominal_height(cfg_eval)) if is_random_stairs else float(cfg_eval.stair_step_height)
    )
    world_cfg.stair_step_width = float(cfg_eval.stair_step_width)
    world_cfg.stair_num_steps_up = int(cfg_eval.stair_num_steps_up)
    world_cfg.stair_num_steps_down = int(cfg_eval.stair_num_steps_down)
    world = DogWorld(world_cfg)

    _add_unmodeled_payload_to_world(world, cfg_eval)

    if terrain == "procedural":
        _add_procedural_terrain_to_world(world, cfg_eval, seed=seed)
        return world
    if is_random_stairs:
        step_depth = float(getattr(world_cfg, "stair_step_depth", 3.0))
        _add_random_stairs_to_world(world, cfg_eval, step_depth=step_depth, seed=seed)
        return world

    # DogWorld builds the first set from world_cfg; add subsequent sets here.
    num_sets = int(max(1, cfg_eval.stair_num_sets))
    if num_sets > 1:
        single_set_length = (
            float(cfg_eval.stair_num_steps_up * cfg_eval.stair_step_width)
            + float(cfg_eval.stair_num_steps_down * cfg_eval.stair_step_width)
        )
        current_x = float(cfg_eval.stair_start_x + single_set_length + cfg_eval.stair_gap_distance)
        for set_idx in range(1, num_sets):
            _add_stair_set_to_world(
                world=world,
                set_idx=set_idx,
                start_x=current_x,
                step_height=float(cfg_eval.stair_step_height),
                step_width=float(cfg_eval.stair_step_width),
                step_depth=float(getattr(world_cfg, "stair_step_depth", 3.0)),
                num_steps_up=int(cfg_eval.stair_num_steps_up),
                num_steps_down=int(cfg_eval.stair_num_steps_down),
            )
            current_x += single_set_length + float(cfg_eval.stair_gap_distance)
        print(
            f"[compare] Added stair sets: total_sets={num_sets}, gap_m={float(cfg_eval.stair_gap_distance):.2f}, "
            f"up={int(cfg_eval.stair_num_steps_up)}, down={int(cfg_eval.stair_num_steps_down)}"
        )

    return world


def _compute_course_geometry(cfg_eval: EvalConfig, terrain: str) -> Dict[str, object]:
    """Compute geometry for stair courses or procedural terrain."""
    if terrain == "procedural":
        return {
            "terrain": "procedural",
            "course_end_x": float(cfg_eval.procedural_narrow_success_x),
            "procedural_narrow_success_x": float(cfg_eval.procedural_narrow_success_x),
            "survival_success_duration_steps": int(cfg_eval.survival_success_duration_steps),
            "stagnation_activation_x": -0.2,
        }
    if terrain == "random_stairs":
        nominal_step_height = float(_random_stair_nominal_height(cfg_eval))
    else:
        nominal_step_height = float(cfg_eval.stair_step_height)
    stair_top_height = float(cfg_eval.stair_num_steps_up * nominal_step_height)
    single_set_length = cfg_eval.stair_num_steps_up * cfg_eval.stair_step_width + cfg_eval.stair_num_steps_down * cfg_eval.stair_step_width
    
    # For multi-set courses, compute end positions
    set_positions = []
    current_x = cfg_eval.stair_start_x
    
    for set_idx in range(int(cfg_eval.stair_num_sets)):
        stair_up_end_x = current_x + cfg_eval.stair_num_steps_up * cfg_eval.stair_step_width
        stair_down_end_x = stair_up_end_x + cfg_eval.stair_num_steps_down * cfg_eval.stair_step_width
        set_positions.append({"up_end_x": stair_up_end_x, "down_end_x": stair_down_end_x})
        current_x = stair_down_end_x + cfg_eval.stair_gap_distance
    
    # Course ends after the final descent
    course_end_x = set_positions[-1]["down_end_x"] if set_positions else cfg_eval.stair_start_x + single_set_length
    descend_flat_z_margin_m = max(0.08, nominal_step_height + 0.04)
    descend_down_end_x_margin_m = max(0.05, cfg_eval.descend_down_end_x_margin_m_scale * cfg_eval.stair_step_width)
    
    return {
        "stair_up_end_x": float(set_positions[0]["up_end_x"]) if set_positions else float(cfg_eval.stair_start_x + cfg_eval.stair_num_steps_up * cfg_eval.stair_step_width),
        "course_end_x": float(course_end_x),
        "stair_top_height": float(stair_top_height),
        "descend_flat_z_margin_m": float(descend_flat_z_margin_m),
        "descend_down_end_x_margin_m": float(descend_down_end_x_margin_m),
        "set_positions": set_positions,  # List of dicts with up_end_x and down_end_x for each set
        "num_sets": int(cfg_eval.stair_num_sets),
        "stagnation_activation_x": float(cfg_eval.stair_start_x - 0.3),
    }


def _configure_case(runner: DogRunner, case_name: str) -> None:
    def _apply_reflex_mode(trigger_mode, use_next_steps_enhancement: bool) -> None:
        # Keep config and live ESD object in sync for deterministic per-case behavior.
        cfg.simulation_params["reflex_trigger_mode"] = trigger_mode
        cfg.simulation_params["reflex_next_steps_height_enhancement"] = bool(use_next_steps_enhancement)
        try:
            esd = runner.mpc_controller.wb_interface.esd
            esd.trigger_mode = trigger_mode
            esd.activated = bool(trigger_mode is not False)
            esd.use_reflex_next_steps_height_enhancement = bool(use_next_steps_enhancement)
        except Exception:
            pass

    def _reset_nominal_gait_state() -> None:
        nominal_step_height = float(cfg.simulation_params.get("step_height", 0.12))
        nominal_swing_time = float(cfg.simulation_params.get("adaptive_swing_time", 0.20))
        
        runner.mpc_controller.enable_rl_step_height(False)
        runner.mpc_controller.set_step_height_residual(0.0)
        runner.mpc_controller.wb_interface.step_height = float(runner.mpc_controller._nominal_step_height)
        runner.mpc_controller.set_adaptive_swing_time(nominal_swing_time)
        runner.mpc_controller.set_rl_observation_context(0.0, swing_time=nominal_swing_time)
        runner.mpc_controller.wb_interface.adaptive_swing_time = nominal_swing_time
        runner.mpc_controller.wb_interface.frg.swing_time = nominal_swing_time


    if case_name == "mpc_only":
        runner.rl_policy_enabled = False
        _apply_reflex_mode("tracking", True)
        _reset_nominal_gait_state()
    elif case_name == "mpc_rl":
        if runner.rl_actor is None:
            reason = getattr(runner, "rl_actor_init_error", None)
            suffix = f" Load error: {reason}" if reason else ""
            raise RuntimeError(
                "MPC+RL case requested but RL checkpoint was not loaded. "
                "Set simulation_params['rl_residual_checkpoint'] to a valid *.pt path."
                + suffix
            )
        runner.rl_policy_enabled = True
        _apply_reflex_mode(False, False)
        try:
            runner.mpc_controller.enable_rl_step_height(True)
            runner.mpc_controller.set_step_height_residual(0.0)
        except Exception:
            pass
    else:
        raise ValueError(f"Unknown case: {case_name}")


def _reset_episode(runner: DogRunner, cfg_eval: EvalConfig) -> None:
    from src.dog.dog_runner import SimulationState

    # Disable auto-activation in this scripted evaluator so episode state
    # transitions are deterministic and tied only to this script.
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

    # Keep INIT/standing for a short recovery window before enabling ACTIVE,
    # matching the training env reset strategy.
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

    # Activate MPC control and set nominal command (ramped inside episode).
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
    runner: DogRunner,
    cfg_eval: EvalConfig,
    geom: Dict[str, object],
    terrain: str,
    headless: bool,
) -> Tuple[EpisodeMetrics, Dict[str, np.ndarray]]:
    sim_dt = float(runner.sim_dt)

    step = 0
    start_z = None
    course_entry_step = -1
    course_complete_step = -1
    climbed_peak = False
    descended_to_flat = False
    success_post_steps = 0
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
    
    # Stagnation detection: track recent X positions
    position_history: List[float] = []
    stagnation_window = int(cfg_eval.stagnation_check_window_steps)
    stagnation_threshold = float(cfg_eval.stagnation_distance_threshold)

    while step < cfg_eval.episode_length_steps:
        # Match training-style command ramp to avoid startup transients.
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

        if not headless and (step % cfg_eval.render_every_n_steps == 0):
            runner.stage.world.step(render=True)
        else:
            runner.stage.world.step(render=False)
        runner.post_physics()

        snap = runner.mpc_controller.get_rl_state_snapshot()
        if isinstance(snap, dict):
            pos = np.asarray(snap.get("position", np.zeros(3)), dtype=np.float64)
            lin = np.asarray(snap.get("linear_velocity", np.zeros(3)), dtype=np.float64)
            ref_lin_nom = np.asarray(snap.get("ref_base_lin_vel", np.zeros(3)), dtype=np.float64)

            if start_z is None:
                start_z = float(pos[2])

            cmd_speed_nom = float(np.linalg.norm(ref_lin_nom[0:2]))
            act_speed_nom = float(np.linalg.norm(lin[0:2]))
            tracking_error = abs(act_speed_nom - cmd_speed_nom)

            cmd_speeds.append(cmd_speed_nom)
            act_speeds.append(act_speed_nom)
            track_errors.append(float(tracking_error))

            base_x = float(pos[0])
            base_z = float(pos[2])
            base_x_trace.append(base_x)
            base_z_trace.append(base_z)

            euler = np.asarray(snap.get("base_ori_euler_xyz", np.zeros(3)), dtype=np.float64).reshape(-1)
            if euler.size >= 3:
                roll = float(np.arctan2(np.sin(euler[0]), np.cos(euler[0])))
                pitch = float(np.arctan2(np.sin(euler[1]), np.cos(euler[1])))
                yaw = float(np.arctan2(np.sin(euler[2]), np.cos(euler[2])))
                roll_error_rad.append(abs(roll))
                pitch_error_rad.append(abs(pitch))
                yaw_error_rad.append(abs(yaw))

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

                # Get joint velocities directly from the RL snapshot
                joint_vel_raw = snap.get("joint_velocities", None)
                if joint_vel_raw is not None:
                    joint_vel_arr = np.asarray(joint_vel_raw, dtype=np.float64).reshape(-1)
                else:
                    joint_vel_arr = np.array([])
                
                if torques_arr.size >= 12 and joint_vel_arr.size >= 12:
                    # mechanical power = sum(|tau * omega|)
                    power = float(np.sum(np.abs(torques_arr[:12] * joint_vel_arr[:12])))
                else:
                    raise RuntimeError("Insufficient torques or joint velocities to calculate power.")

            joint_torque_mean_abs.append(torque_mean)

            # Calculate Cost of Transport (CoT = Power / (mass * g * velocity))
            # Assuming ~15kg for go2 and g=9.81, and include payload mass if configured.
            robot_mass = float(getattr(runner, "robot_mass", 15.0))
            payload_mass = float(getattr(cfg_eval, "payload_mass_kg", 0.0))
            total_mass = robot_mass + max(0.0, payload_mass)
            if act_speed_nom > 0.01:
                cot = power / (total_mass * 9.81 * act_speed_nom)
            else:
                cot = power / (total_mass * 9.81 * 0.05)
            cot_trace.append(cot)

            # Effective step height mirrors DogRunner diagnostics.
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

            # Track position history for stagnation detection
            position_history.append(base_x)
            if len(position_history) > stagnation_window:
                position_history.pop(0)

            # Unified success criteria for both terrains.
            reached_success_x = base_x >= float(cfg_eval.procedural_narrow_success_x)
            if reached_success_x:
                if course_complete_step < 0:
                    course_complete_step = step
                break

            # Check for stagnation: if robot hasn't moved enough in recent steps, terminate
            stagnation_check_min_step = int(max(cfg_eval.command_ramp_steps, cfg_eval.stagnation_check_window_steps))
            if terrain in ("stairs", "random_stairs"):
                in_stagnation_region = bool(base_x >= (cfg_eval.stair_start_x - 0.3))
            else:
                in_stagnation_region = bool(base_x >= float(geom.get("stagnation_activation_x", -0.2)))
            if len(position_history) >= stagnation_window and step >= stagnation_check_min_step and in_stagnation_region:
                displacement = position_history[-1] - position_history[0]
                if displacement < stagnation_threshold:
                    stagnation_terminated = True
                    stagnation_step = int(step)
                    break

            if terrain in ("stairs", "random_stairs"):
                if course_entry_step < 0 and base_x >= cfg_eval.stair_start_x:
                    course_entry_step = step
            else:
                if course_entry_step < 0:
                    course_entry_step = 0

        step += 1

    course_complete = float(1.0 if course_complete_step >= 0 else 0.0)

    completion_time_s = float("nan")
    if course_complete_step >= 0:
        # Count from episode timestep 0 to strict course-completion timestep.
        completion_steps = int(course_complete_step + 1)
        completion_time_s = float(completion_steps * sim_dt)

    course_success = float(course_complete)
    goal_crossed = float(course_complete)

    metrics = EpisodeMetrics(
        terrain=terrain,
        command_forward=float(cfg_eval.command_forward),
        stair_step_height=float(cfg_eval.stair_step_height),
        case="",
        episode_idx=-1,
        course_success=course_success,
        course_complete=course_complete,
        goal_crossed=goal_crossed,
        completion_time_s=completion_time_s,
        strict_completion_time_s=completion_time_s,
        mean_tracking_error=float(np.mean(track_errors)) if track_errors else float("nan"),
        mean_cmd_speed_nominal=float(np.mean(cmd_speeds)) if cmd_speeds else float("nan"),
        mean_actual_speed_nominal=float(np.mean(act_speeds)) if act_speeds else float("nan"),
        mean_swing_time_s=float(np.mean(swing_time_s)) if swing_time_s else float("nan"),
        mean_cot=float(np.mean(cot_trace)) if cot_trace else float("nan"),
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


def _plot_results(
    out_png: str,
    rows: List[EpisodeMetrics],
    traces_by_case: Dict[str, List[Dict[str, np.ndarray]]],
    sim_dt: float,
) -> None:
    import matplotlib.pyplot as plt

    mpc = [r for r in rows if r.case == "mpc_only"]
    rl = [r for r in rows if r.case == "mpc_rl"]

    fig, axes = plt.subplots(3, 1, figsize=(11, 11), constrained_layout=True)

    mpc_case_traces = traces_by_case.get("mpc_only", [])
    rl_case_traces = traces_by_case.get("mpc_rl", [])
    mpc_cmd = _aggregate_case_trace([t.get("cmd_speed", np.zeros(0, dtype=np.float32)) for t in mpc_case_traces])
    mpc_act = _aggregate_case_trace([t.get("actual_speed", np.zeros(0, dtype=np.float32)) for t in mpc_case_traces])
    rl_cmd = _aggregate_case_trace([t.get("cmd_speed", np.zeros(0, dtype=np.float32)) for t in rl_case_traces])
    rl_act = _aggregate_case_trace([t.get("actual_speed", np.zeros(0, dtype=np.float32)) for t in rl_case_traces])
    mpc_swing = _aggregate_case_trace([t.get("swing_time_s", np.zeros(0, dtype=np.float32)) for t in mpc_case_traces])
    rl_swing = _aggregate_case_trace([t.get("swing_time_s", np.zeros(0, dtype=np.float32)) for t in rl_case_traces])
    mpc_swing_foot_z = _aggregate_case_trace([t.get("swing_foot_z", np.zeros(0, dtype=np.float32)) for t in mpc_case_traces])
    rl_swing_foot_z = _aggregate_case_trace([t.get("swing_foot_z", np.zeros(0, dtype=np.float32)) for t in rl_case_traces])

    max_len = max(
        int(mpc_cmd.shape[0]),
        int(mpc_act.shape[0]),
        int(rl_cmd.shape[0]),
        int(rl_act.shape[0]),
        int(mpc_swing.shape[0]),
        int(rl_swing.shape[0]),
        int(mpc_swing_foot_z.shape[0]),
        int(rl_swing_foot_z.shape[0]),
    )
    if max_len > 0:
        t = np.arange(max_len, dtype=np.float32) * float(sim_dt)

        def _pad_to(a: np.ndarray, n: int) -> np.ndarray:
            if a.shape[0] >= n:
                return a[:n]
            if a.shape[0] == 0:
                return np.zeros(n, dtype=np.float32)
            return np.pad(a, (0, n - a.shape[0]), mode="edge")

        mpc_cmd = _pad_to(mpc_cmd, max_len)
        mpc_act = _pad_to(mpc_act, max_len)
        rl_cmd = _pad_to(rl_cmd, max_len)
        rl_act = _pad_to(rl_act, max_len)
        mpc_swing = _pad_to(mpc_swing, max_len)
        rl_swing = _pad_to(rl_swing, max_len)
        mpc_swing_foot_z = _pad_to(mpc_swing_foot_z, max_len)
        rl_swing_foot_z = _pad_to(rl_swing_foot_z, max_len)

        axes[0].plot(t, mpc_cmd, "--", color="#17becf", label="MPC-only commanded")
        axes[0].plot(t, mpc_act, "-", color="#1f77b4", label="MPC-only actual")
        axes[0].plot(t, rl_cmd, "--", color="#bcbd22", label="MPC+RL commanded")
        axes[0].plot(t, rl_act, "-", color="#d62728", label="MPC+RL actual")
    axes[0].set_title("Commanded/Actual Speed vs Time")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Speed [m/s]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    mpc_by_ep = {int(r.episode_idx): float(r.completion_time_s) for r in mpc}
    rl_by_ep = {int(r.episode_idx): float(r.completion_time_s) for r in rl}
    episode_indices = sorted(set(mpc_by_ep.keys()) | set(rl_by_ep.keys()))
    if episode_indices:
        x = np.arange(len(episode_indices), dtype=np.float32)
        bar_w = 0.38
        mpc_vals = np.asarray([mpc_by_ep.get(ep, float("nan")) for ep in episode_indices], dtype=np.float32)
        rl_vals = np.asarray([rl_by_ep.get(ep, float("nan")) for ep in episode_indices], dtype=np.float32)

        mpc_ok = np.isfinite(mpc_vals)
        rl_ok = np.isfinite(rl_vals)
        if np.any(mpc_ok):
            axes[1].bar(x[mpc_ok] - 0.5 * bar_w, mpc_vals[mpc_ok], width=bar_w, color="#1f77b4", alpha=0.8, label="MPC-only")
        if np.any(rl_ok):
            axes[1].bar(x[rl_ok] + 0.5 * bar_w, rl_vals[rl_ok], width=bar_w, color="#d62728", alpha=0.8, label="MPC+RL")

        axes[1].set_xticks(x)
        axes[1].set_xticklabels([str(ep) for ep in episode_indices])

    axes[1].set_title("Completion-Time Histogram by Episode")
    axes[1].set_xlabel("Episode number")
    axes[1].set_ylabel("Completion time [s] (NaN = not completed)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    mpc_swing_by_ep = {int(r.episode_idx): float(r.mean_swing_time_s) for r in mpc}
    rl_swing_by_ep = {int(r.episode_idx): float(r.mean_swing_time_s) for r in rl}
    if episode_indices:
        x = np.arange(len(episode_indices), dtype=np.float32)
        bar_w = 0.38
        mpc_vals = np.asarray([mpc_swing_by_ep.get(ep, float("nan")) for ep in episode_indices], dtype=np.float32)
        rl_vals = np.asarray([rl_swing_by_ep.get(ep, float("nan")) for ep in episode_indices], dtype=np.float32)

        mpc_ok = np.isfinite(mpc_vals)
        rl_ok = np.isfinite(rl_vals)
        if np.any(mpc_ok):
            axes[2].bar(x[mpc_ok] - 0.5 * bar_w, mpc_vals[mpc_ok], width=bar_w, color="#1f77b4", alpha=0.8, label="MPC-only")
        if np.any(rl_ok):
            axes[2].bar(x[rl_ok] + 0.5 * bar_w, rl_vals[rl_ok], width=bar_w, color="#d62728", alpha=0.8, label="MPC+RL")

        axes[2].set_xticks(x)
        axes[2].set_xticklabels([str(ep) for ep in episode_indices])

    axes[2].plot(t, mpc_swing_foot_z, "-", color="#1f77b4", linewidth=2.0, label="MPC-only swing foot")
    axes[2].plot(t, rl_swing_foot_z, "-", color="#d62728", linewidth=2.0, label="MPC+RL swing foot")
    axes[2].set_title("Swing-Foot Z Height vs Time")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("Swing foot Z [m]")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best")

    fig.savefig(out_png, dpi=140)


def _aggregate_case_trace(traces: List[np.ndarray]) -> np.ndarray:
    if not traces:
        return np.zeros(0, dtype=np.float32)
    max_len = max(int(t.shape[0]) for t in traces)
    if max_len <= 0:
        return np.zeros(0, dtype=np.float32)
    arr = np.full((len(traces), max_len), np.nan, dtype=np.float32)
    for i, t in enumerate(traces):
        n = int(t.shape[0])
        if n > 0:
            arr[i, :n] = t
    return np.asarray(np.nanmean(arr, axis=0), dtype=np.float32)


def _plot_speed_and_step_height_profiles(out_png: str, traces_by_case: Dict[str, List[Dict[str, np.ndarray]]], sim_dt: float) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), constrained_layout=True)
    case_specs = [("mpc_only", "MPC-only", "#1f77b4"), ("mpc_rl", "MPC+RL", "#d62728")]

    for ax, (case_key, case_label, color) in zip(axes, case_specs):
        case_traces = traces_by_case.get(case_key, [])
        cmd = _aggregate_case_trace([t.get("cmd_speed", np.zeros(0, dtype=np.float32)) for t in case_traces])
        act = _aggregate_case_trace([t.get("actual_speed", np.zeros(0, dtype=np.float32)) for t in case_traces])
        step_h = _aggregate_case_trace([t.get("step_height_eff", np.zeros(0, dtype=np.float32)) for t in case_traces])
        swing_t = _aggregate_case_trace([t.get("swing_time_s", np.zeros(0, dtype=np.float32)) for t in case_traces])

        max_len = max(int(cmd.shape[0]), int(act.shape[0]), int(step_h.shape[0]), int(swing_t.shape[0]))
        if max_len <= 0:
            ax.set_title(f"{case_label}: no trace data")
            ax.grid(True, alpha=0.3)
            continue

        t = np.arange(max_len, dtype=np.float32) * float(sim_dt)
        if cmd.shape[0] < max_len:
            cmd = np.pad(cmd, (0, max_len - cmd.shape[0]), mode="edge") if cmd.shape[0] > 0 else np.zeros(max_len, dtype=np.float32)
        if act.shape[0] < max_len:
            act = np.pad(act, (0, max_len - act.shape[0]), mode="edge") if act.shape[0] > 0 else np.zeros(max_len, dtype=np.float32)
        if step_h.shape[0] < max_len:
            step_h = np.pad(step_h, (0, max_len - step_h.shape[0]), mode="edge") if step_h.shape[0] > 0 else np.zeros(max_len, dtype=np.float32)
        if swing_t.shape[0] < max_len:
            swing_t = np.pad(swing_t, (0, max_len - swing_t.shape[0]), mode="edge") if swing_t.shape[0] > 0 else np.zeros(max_len, dtype=np.float32)

        ax.plot(t, cmd, "--", color="#2ca02c", linewidth=1.8, label="commanded speed")
        ax.plot(t, act, "-", color=color, linewidth=2.0, label="actual speed")
        ax.set_ylabel("Speed [m/s]")
        ax.set_title(f"{case_label}: Commanded/Actual Speed + Effective Step Height + Swing Time")
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
    terrain: str,
    command_forward: float,
    traces_by_case: Dict[str, List[Dict[str, np.ndarray]]],
    sim_dt: float,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(12, 5), constrained_layout=True)
    case_specs = [("mpc_only", "MPC-only", "#1f77b4"), ("mpc_rl", "MPC+RL", "#d62728")]

    max_len = 0
    traces = {}
    for case_key, _, _ in case_specs:
        case_traces = traces_by_case.get(case_key, [])
        cmd = _aggregate_case_trace([t.get("cmd_speed", np.zeros(0, dtype=np.float32)) for t in case_traces])
        act = _aggregate_case_trace([t.get("actual_speed", np.zeros(0, dtype=np.float32)) for t in case_traces])
        traces[case_key] = (cmd, act)
        max_len = max(max_len, int(cmd.shape[0]), int(act.shape[0]))

    if max_len <= 0:
        ax.set_title(f"{terrain} | target speed {command_forward:.2f} m/s: no trace data")
        ax.grid(True, alpha=0.3)
        fig.savefig(out_png, dpi=140)
        return

    t = np.arange(max_len, dtype=np.float32) * float(sim_dt)
    for case_key, label, color in case_specs:
        cmd, act = traces.get(case_key, (np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)))
        if cmd.shape[0] < max_len:
            cmd = np.pad(cmd, (0, max_len - cmd.shape[0]), mode="edge") if cmd.shape[0] > 0 else np.zeros(max_len, dtype=np.float32)
        if act.shape[0] < max_len:
            act = np.pad(act, (0, max_len - act.shape[0]), mode="edge") if act.shape[0] > 0 else np.zeros(max_len, dtype=np.float32)
        ax.plot(t, cmd, "--", color=color, alpha=0.45, linewidth=1.5)
        ax.plot(t, act, "-", color=color, linewidth=2.2, label=f"{label} actual")

    ax.axhline(float(command_forward), color="#2ca02c", linestyle="--", linewidth=1.6, label="target speed")
    ax.set_title(f"{terrain} | target speed {command_forward:.2f} m/s: velocity tracking")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Speed [m/s]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(out_png, dpi=140)


def _plot_base_z_by_speed(
    out_png: str,
    terrain: str,
    command_forward: float,
    traces_by_case: Dict[str, List[Dict[str, np.ndarray]]],
    sim_dt: float,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(12, 5), constrained_layout=True)
    case_specs = [("mpc_only", "MPC-only", "#1f77b4"), ("mpc_rl", "MPC+RL", "#d62728")]

    max_len = 0
    traces = {}
    for case_key, _, _ in case_specs:
        case_traces = traces_by_case.get(case_key, [])
        base_z = _aggregate_case_trace([t.get("base_z", np.zeros(0, dtype=np.float32)) for t in case_traces])
        traces[case_key] = base_z
        max_len = max(max_len, int(base_z.shape[0]))

    if max_len <= 0:
        ax.set_title(f"{terrain} | target speed {command_forward:.2f} m/s: base z vs time (no data)")
        ax.grid(True, alpha=0.3)
        fig.savefig(out_png, dpi=140)
        return

    t = np.arange(max_len, dtype=np.float32) * float(sim_dt)
    for case_key, label, color in case_specs:
        base_z = traces.get(case_key, np.zeros(0, dtype=np.float32))
        if base_z.shape[0] < max_len:
            base_z = (
                np.pad(base_z, (0, max_len - base_z.shape[0]), mode="edge")
                if base_z.shape[0] > 0
                else np.zeros(max_len, dtype=np.float32)
            )
        ax.plot(t, base_z, "-", color=color, linewidth=2.0, label=label)

    ax.set_title(f"{terrain} | target speed {command_forward:.2f} m/s: base z vs time")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Base z [m]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(out_png, dpi=140)


def _plot_success_by_speed(
    out_png: str,
    terrain: str,
    command_forward: float,
    rows: List[EpisodeMetrics],
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 5), constrained_layout=True)
    case_specs = [("mpc_only", "MPC-only", "#1f77b4"), ("mpc_rl", "MPC+RL", "#d62728")]

    labels = []
    success_rates = []
    success_counts = []
    total_counts = []
    colors = []
    for case_key, label, color in case_specs:
        case_rows = [r for r in rows if r.case == case_key]
        total = len(case_rows)
        success_count = int(sum(1 for r in case_rows if float(r.course_success) > 0.5))
        success_rate = float(success_count / total) if total > 0 else float("nan")
        labels.append(label)
        success_rates.append(success_rate)
        success_counts.append(success_count)
        total_counts.append(total)
        colors.append(color)

    x = np.arange(len(labels), dtype=np.float32)
    bars = ax.bar(x, success_rates, color=colors, alpha=0.88)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Success rate")
    ax.set_title(f"{terrain} | target speed {command_forward:.2f} m/s: successful episodes by mode")
    ax.grid(True, axis="y", alpha=0.3)

    for bar, success_count, total in zip(bars, success_counts, total_counts):
        height = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            min(1.02, height + 0.04),
            f"{success_count}/{total}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.savefig(out_png, dpi=140)


def _plot_metric_by_speed(
    out_png: str,
    terrain: str,
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
        ax.set_title(f"{terrain} | {title}: no speeds")
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
        ax.set_title(f"{terrain} | {title}: no data")
    else:
        ax.set_title(f"{terrain} | {title}")
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
    terrain: str,
    speed_values: List[float],
    series: Dict[str, List[int]],
    title: str,
    x_label: str = "Target speed [m/s]",
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(9.5, 5), constrained_layout=True)
    if not speed_values:
        ax.set_title(f"{terrain} | {title}: no speeds")
        fig.savefig(out_png, dpi=140)
        return

    if not series:
        ax.set_title(f"{terrain} | {title}: no data")
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
        ax.set_title(f"{terrain} | {title}: no data")
        fig.savefig(out_png, dpi=140)
        return

    ax.set_title(f"{terrain} | {title}")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Successful episodes")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.2f}" for v in speed_values])
    ax.set_ylim(0.0, float(max(max_count, 1)) + 0.6)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(out_png, dpi=140)


def _plot_success_summary_by_speed_and_height(
    out_png: str,
    terrain: str,
    speed_values: List[float],
    stair_height_values: List[float],
    speed_series_by_mode: Dict[str, Dict[str, List[float]]],
    height_series_by_mode: Dict[str, Dict[str, List[float]]],
    ylabel: str,
    title: str,
    value_format: str = "{:.2f}",
) -> None:
    import matplotlib.pyplot as plt

    mode_labels = list(speed_series_by_mode.keys())
    if not mode_labels:
        mode_labels = list(height_series_by_mode.keys())

    if not mode_labels:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4), constrained_layout=True)
        ax.set_title(f"{terrain} | {title}: no data")
        fig.savefig(out_png, dpi=140)
        return

    fig, axes = plt.subplots(len(mode_labels), 2, figsize=(13.5, 4.8 * len(mode_labels)), constrained_layout=True)
    if len(mode_labels) == 1:
        axes = np.asarray([axes])

    x_speed = np.asarray(speed_values, dtype=np.float32)
    x_height = np.asarray(stair_height_values, dtype=np.float32)

    for row_idx, mode_label in enumerate(mode_labels):
        ax_speed = axes[row_idx, 0]
        ax_height = axes[row_idx, 1]

        mode_speed_series = speed_series_by_mode.get(mode_label, {})
        mode_height_series = height_series_by_mode.get(mode_label, {})

        # --- Speed plot: grouped bars per speed, series keyed by height label ---
        speed_labels = [k for k, v in mode_speed_series.items() if len(v) == len(speed_values) and np.any(np.isfinite(v))]
        n_speed_series = max(1, len(speed_labels))
        x_idx = np.arange(len(speed_values), dtype=np.float32)
        width = 0.8 / float(n_speed_series)
        if speed_labels:
            for i, height_label in enumerate(speed_labels):
                y = np.asarray(mode_speed_series[height_label], dtype=np.float32)
                # replace NaN with 0 for plotting but make masked where NaN
                y_plot = np.nan_to_num(y, nan=0.0)
                bars = ax_speed.bar(x_idx + (i - (n_speed_series - 1) / 2.0) * width, y_plot, width=width, label=height_label)
                # dim bars where values were NaN
                for j, bar in enumerate(bars):
                    if not np.isfinite(y[j]):
                        bar.set_alpha(0.18)
            ax_speed.set_title(f"{mode_label}: {title} vs speed")
            ax_speed.set_xticks(x_idx)
            ax_speed.set_xticklabels([f"{v:.2f}" for v in speed_values])
            ax_speed.set_xlabel("Target speed [m/s]")
            ax_speed.set_ylabel(ylabel)
            ax_speed.grid(True, alpha=0.3)
            ax_speed.legend(loc="best")
        else:
            ax_speed.set_title(f"{mode_label}: no speed data")
            ax_speed.set_xlabel("Target speed [m/s]")
            ax_speed.set_ylabel(ylabel)
            ax_speed.grid(True, alpha=0.3)

        # --- Height plot: grouped bars per stair height, series keyed by speed label ---
        height_labels = [k for k, v in mode_height_series.items() if len(v) == len(stair_height_values) and np.any(np.isfinite(v))]
        n_height_series = max(1, len(height_labels))
        x_idx_h = np.arange(len(stair_height_values), dtype=np.float32)
        width_h = 0.8 / float(n_height_series)
        if height_labels:
            for i, speed_label in enumerate(height_labels):
                y = np.asarray(mode_height_series[speed_label], dtype=np.float32)
                y_plot = np.nan_to_num(y, nan=0.0)
                bars = ax_height.bar(x_idx_h + (i - (n_height_series - 1) / 2.0) * width_h, y_plot, width=width_h, label=speed_label)
                for j, bar in enumerate(bars):
                    if not np.isfinite(y[j]):
                        bar.set_alpha(0.18)
            ax_height.set_title(f"{mode_label}: {title} vs stair height")
            ax_height.set_xticks(x_idx_h)
            ax_height.set_xticklabels([value_format.format(v) for v in stair_height_values])
            ax_height.set_xlabel("Stair step height [m]")
            ax_height.set_ylabel(ylabel)
            ax_height.grid(True, alpha=0.3)
            ax_height.legend(loc="best")
        else:
            ax_height.set_title(f"{mode_label}: no stair-height data")
            ax_height.set_xlabel("Stair step height [m]")
            ax_height.set_ylabel(ylabel)
            ax_height.grid(True, alpha=0.3)

    fig.savefig(out_png, dpi=140)


def _plot_attitude_stability_by_speed(
    out_png: str,
    terrain: str,
    speed_values: List[float],
    roll_series: Dict[str, List[float]],
    pitch_series: Dict[str, List[float]],
    yaw_series: Dict[str, List[float]],
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 10.5), constrained_layout=True)
    if not speed_values:
        axes[0].set_title(f"{terrain} | Attitude stability: no speeds")
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
            ax.set_title(f"{terrain} | {title}: no data")
        else:
            ax.set_title(f"{terrain} | {title}")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if any_plotted:
            ax.legend(loc="best")

    _plot_axis(axes[0], roll_series, "Mean roll tracking error vs speed", "Mean |roll| [rad]")
    _plot_axis(axes[1], pitch_series, "Mean pitch tracking error vs speed", "Mean |pitch| [rad]")
    _plot_axis(axes[2], yaw_series, "Mean yaw tracking error vs speed", "Mean |yaw| [rad]")
    axes[-1].set_xlabel("Target speed [m/s]")
    fig.savefig(out_png, dpi=140)


def _safe_nanmean(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _build_speed_sweep(speed_min: float, speed_max: float, speed_step: float) -> List[float]:
    if speed_step <= 0.0:
        raise ValueError("--speed-step must be greater than zero")
    if speed_max < speed_min:
        raise ValueError("--speed-max must be greater than or equal to --speed-min")

    speeds: List[float] = []
    current = float(speed_min)
    epsilon = 1e-9
    while current <= float(speed_max) + epsilon:
        speeds.append(float(np.round(current, 6)))
        current += float(speed_step)

    if not speeds:
        speeds.append(float(np.round(speed_min, 6)))
    if abs(speeds[-1] - float(speed_max)) > 1e-6:
        speeds.append(float(np.round(speed_max, 6)))
    return speeds


def _build_stair_height_sweep(stair_height_min: float, stair_height_max: float, stair_height_step: float) -> List[float]:
    if stair_height_step <= 0.0:
        raise ValueError("--stair-height-step must be greater than zero")
    if stair_height_max < stair_height_min:
        raise ValueError("--stair-height-max must be greater than or equal to --stair-height-min")

    heights: List[float] = []
    current = float(stair_height_min)
    epsilon = 1e-9
    while current <= float(stair_height_max) + epsilon:
        heights.append(float(np.round(current, 6)))
        current += float(stair_height_step)

    if not heights:
        heights.append(float(np.round(stair_height_min, 6)))
    if abs(heights[-1] - float(stair_height_max)) > 1e-6:
        heights.append(float(np.round(stair_height_max, 6)))
    return heights


def _speed_tag(speed: float) -> str:
    return f"{float(speed):.2f}".replace("-", "m").replace(".", "p")


def _worker_process(
    worker_id: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    args: argparse.Namespace,
    cfg_eval: EvalConfig,
    ckpt_abs: str,
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

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": args.headless})

    from src.dog.dog_runner import DogRunner
    import omni.usd

    current_terrain = None
    current_stair_height = None
    current_seed = None
    world = None
    runner = None
    runner_case_mode = None

    def _build_runner(case_name: str) -> DogRunner:
        if case_name == "mpc_only":
            # Strict isolation: do not load RL checkpoint at all for the baseline case.
            return DogRunner(
                world,
                rl_residual_checkpoint=None,
                rl_residual_device=args.device,
                rl_residual_deterministic=not args.stochastic,
                rl_residual_warmup_steps=int(cfg.simulation_params.get("rl_residual_warmup_steps", 120)),
                rl_residual_ramp_steps=int(cfg.simulation_params.get("rl_residual_ramp_steps", 400)),
            )

        return DogRunner(
            world,
            rl_residual_checkpoint=ckpt_abs,
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

        terrain = task["terrain"]
        command_forward = float(task["command_forward"])
        stair_step_height = float(task.get("stair_step_height", cfg_eval.stair_step_height))
        ep = task["ep_idx"]
        seed = task["seed"]
        case_order = task["case_order"]
        geom = task["geom"]

        if terrain in ("stairs", "random_stairs"):
            cfg_eval.stair_step_height = float(stair_step_height)

        needs_new_world = current_terrain != terrain
        if terrain == "stairs" and current_stair_height != stair_step_height:
            needs_new_world = True
        if terrain == "random_stairs" and current_seed != seed:
            needs_new_world = True

        if needs_new_world:
            if runner is not None:
                runner.shutdown()
                runner = None
                runner_case_mode = None
            if world is not None:
                omni.usd.get_context().new_stage()

            world = _make_world(
                cfg_eval,
                terrain,
                seed=seed if terrain in ("procedural", "random_stairs") else None,
            )
            current_terrain = terrain
            current_stair_height = stair_step_height if terrain == "stairs" else None
            current_seed = seed if terrain in ("procedural", "random_stairs") else None
        else:
            if terrain == "procedural" and current_seed != seed:
                print(f"[Worker {worker_id}][compare:{terrain}] Re-randomizing procedural terrain for task (ep={ep}, seed={seed})...")
                _add_procedural_terrain_to_world(world, cfg_eval, seed=seed)
                current_seed = seed

        for case_name in case_order:
            try:
                desired_runner_mode = "mpc_only" if case_name == "mpc_only" else "mpc_rl"
                if runner is None or runner_case_mode != desired_runner_mode:
                    if runner is not None:
                        runner.shutdown()
                    runner = _build_runner(case_name)
                    runner_case_mode = desired_runner_mode

                cfg_eval.command_forward = float(command_forward)
                _configure_case(runner, case_name)
                _reset_episode(runner, cfg_eval)
                ep_metrics, ep_traces = _run_episode(runner, cfg_eval, geom, terrain, args.headless)
                ep_metrics.case = case_name
                ep_metrics.episode_idx = ep
                ep_metrics.terrain = terrain
                ep_metrics.command_forward = float(command_forward)
                ep_metrics.stair_step_height = float(stair_step_height)

                result_queue.put(
                    {
                        "type": "result",
                        "metrics": ep_metrics,
                        "traces": ep_traces,
                        "case": case_name,
                        "terrain": terrain,
                        "command_forward": command_forward,
                        "stair_step_height": stair_step_height,
                        "ep_idx": ep,
                        "sim_dt": float(runner.sim_dt),
                    }
                )

                print(
                    f"[Worker {worker_id}][compare:{terrain}:{command_forward:.2f}:h{stair_step_height:.3f}:{case_name}] ep={ep} "
                    f"success={ep_metrics.course_success:.0f} complete={ep_metrics.course_complete:.0f} "
                    f"time_s={ep_metrics.completion_time_s:.3f} "
                    f"time_strict_s={ep_metrics.strict_completion_time_s:.3f} "
                    f"track_err_mean={ep_metrics.mean_tracking_error:.4f} "
                    f"cot_mean={ep_metrics.mean_cot:.4f} "
                    f"steps={ep_metrics.total_steps}"
                )
            except Exception:
                result_queue.put(
                    {
                        "type": "error",
                        "message": traceback.format_exc(),
                        "case": case_name,
                        "terrain": terrain,
                        "command_forward": command_forward,
                        "ep_idx": ep,
                    }
                )

    if runner is not None:
        runner.shutdown()
    simulation_app.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare go2-style runtime behavior for MPC-only vs MPC+RL residual. "
            "Runs MPC-only first, then MPC+RL, with training-consistent course completion metrics."
        )
    )
    parser.add_argument("--mode", choices=["both", "mpc_only", "mpc_rl"], default="both")
    parser.add_argument("--terrain", choices=["stairs", "random_stairs", "procedural", "both"], default="stairs")
    parser.add_argument("--stairs", action="store_true", help="Alias for --terrain stairs")
    parser.add_argument("--random-stairs", action="store_true", help="Alias for --terrain random_stairs")
    parser.add_argument("--procedural", action="store_true", help="Alias for --terrain procedural")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=_HEADLESS_DEFAULT)
    parser.add_argument("--num-actors", type=int, default=1, help="Number of parallel actors.")
    parser.add_argument("--output-dir", type=str, default="analysis")
    parser.add_argument("--speed-min", type=float, default=None, help="Minimum forward speed for sweep.")
    parser.add_argument("--speed-max", type=float, default=None, help="Maximum forward speed for sweep.")
    parser.add_argument("--speed-step", type=float, default=None, help="Step size for speed sweep.")
    parser.add_argument("--stair-height-min", type=float, default=None, help="Minimum stair step height for sweep.")
    parser.add_argument("--stair-height-max", type=float, default=None, help="Maximum stair step height for sweep.")
    parser.add_argument("--stair-height-step", type=float, default=None, help="Step size for stair height sweep.")
    parser.add_argument(
        "--payload-mass",
        type=float,
        default=None,
        help="Extra payload mass in kg (0 disables payload).",
    )
    parser.add_argument(
        "--payload-size",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Payload box size in meters (x y z).",
    )
    parser.add_argument(
        "--payload-offset",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Payload offset from base frame in meters (x y z).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(cfg.simulation_params.get("rl_residual_checkpoint", "")),
        help="Checkpoint for DogRunner internal RL residual policy. Defaults to config simulation_params['rl_residual_checkpoint'].",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default=str(cfg.simulation_params.get("rl_residual_device", "cpu")),
        help="Device for DogRunner internal RL policy.",
    )
    parser.add_argument(
        "--stochastic",
        action=argparse.BooleanOptionalAction,
        default=bool(cfg.simulation_params.get("rl_residual_stochastic", False)),
        help="Use stochastic actor action in DogRunner internal RL policy.",
    )
    args = parser.parse_args()

    # Keep task generation stable across runs so checkpoint comparisons are
    # evaluated on the same episode seeds.
    random.seed(42)
    np.random.seed(42)

    # Resolve terrain aliases if provided.
    terrain_arg_used = "--terrain" in sys.argv
    terrain_flags = [args.stairs, args.random_stairs, args.procedural]
    if sum(1 for flag in terrain_flags if flag) > 1:
        raise ValueError("Cannot pass more than one of --stairs, --random-stairs, --procedural.")
    if args.stairs:
        if terrain_arg_used and args.terrain != "stairs":
            raise ValueError("Use either --terrain or --stairs, not both.")
        args.terrain = "stairs"
    if args.random_stairs:
        if terrain_arg_used and args.terrain != "random_stairs":
            raise ValueError("Use either --terrain or --random-stairs, not both.")
        args.terrain = "random_stairs"
    if args.procedural:
        if terrain_arg_used and args.terrain != "procedural":
            raise ValueError("Use either --terrain or --procedural, not both.")
        args.terrain = "procedural"
    if int(args.num_actors) < 1:
        raise ValueError("--num-actors must be >= 1")

    os.makedirs(args.output_dir, exist_ok=True)
    stamp = int(time.time())
    out_csv = os.path.join(args.output_dir, f"go2_mpc_vs_mpc_rl_summary_{stamp}.csv")
    cfg_eval = EvalConfig(episodes_per_case=int(args.episodes))
    if args.speed_min is not None:
        cfg_eval.speed_min = float(args.speed_min)
    if args.speed_max is not None:
        cfg_eval.speed_max = float(args.speed_max)
    if args.speed_step is not None:
        cfg_eval.speed_step = float(args.speed_step)
    if args.stair_height_min is not None:
        cfg_eval.stair_step_height_min = float(args.stair_height_min)
    if args.stair_height_max is not None:
        cfg_eval.stair_step_height_max = float(args.stair_height_max)
    if args.stair_height_step is not None:
        cfg_eval.stair_step_height_step = float(args.stair_height_step)
    if args.payload_mass is not None:
        cfg_eval.payload_mass_kg = float(max(0.0, args.payload_mass))
    if args.payload_size is not None:
        cfg_eval.payload_size_xyz_m = tuple(float(v) for v in args.payload_size)
    if args.payload_offset is not None:
        cfg_eval.payload_offset_xyz_m = tuple(float(v) for v in args.payload_offset)

    speed_values = _build_speed_sweep(cfg_eval.speed_min, cfg_eval.speed_max, cfg_eval.speed_step)
    stair_height_values = _build_stair_height_sweep(
        cfg_eval.stair_step_height_min,
        cfg_eval.stair_step_height_max,
        cfg_eval.stair_step_height_step,
    )

    ckpt_abs = ""
    if args.mode in ("both", "mpc_rl"):
        ckpt = str(args.checkpoint).strip()
        if not ckpt:
            raise ValueError("No checkpoint configured. Set --checkpoint or simulation_params['rl_residual_checkpoint'] in config.py")
        ckpt_expanded = os.path.expanduser(ckpt)
        if os.path.isabs(ckpt_expanded):
            ckpt_abs = os.path.abspath(ckpt_expanded)
        else:
            ckpt_abs = os.path.abspath(os.path.join(REPO_ROOT, ckpt_expanded))
            if not os.path.exists(ckpt_abs):
                # Fallback to cwd semantics for backwards compatibility.
                ckpt_abs = os.path.abspath(ckpt_expanded)
        if not os.path.exists(ckpt_abs):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_abs}")

    all_rows: List[EpisodeMetrics] = []
    case_errors: List[str] = []
    if args.mode == "both":
        case_order = ("mpc_only", "mpc_rl")
    elif args.mode == "mpc_only":
        case_order = ("mpc_only",)
    else:
        case_order = ("mpc_rl",)

    if args.terrain == "both":
        terrain_order = ("stairs", "procedural")
    else:
        terrain_order = (args.terrain,)

    tasks = []
    random_stair_nominal = _random_stair_nominal_height(cfg_eval)
    for terrain in terrain_order:
        if terrain == "stairs":
            terrain_height_values = stair_height_values
        elif terrain == "random_stairs":
            terrain_height_values = [float(random_stair_nominal)]
        else:
            terrain_height_values = [float(cfg_eval.stair_step_height)]
        for stair_step_height in terrain_height_values:
            task_cfg = replace(cfg_eval, stair_step_height=float(stair_step_height))
            geom = _compute_course_geometry(task_cfg, terrain)
            
            for command_forward in speed_values:
                for ep in range(cfg_eval.episodes_per_case):
                    seed = int(np.random.randint(0, 2**31 - 1))
                    tasks.append(
                        {
                            "terrain": terrain,
                            "command_forward": float(command_forward),
                            "stair_step_height": float(stair_step_height),
                            "ep_idx": ep,
                            "seed": seed,
                            "case_order": case_order,
                            "geom": geom,
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
    for i in range(num_workers):
        p = ctx.Process(target=_worker_process, args=(i, task_queue, result_queue, args, cfg_eval, ckpt_abs))
        p.start()
        workers.append(p)
        time.sleep(2.0)

    expected_results = len(tasks) * len(case_order)
    expected_results = len(tasks) * len(case_order) * len(speed_values)
    results_collected = 0

    terrain_rows_by_terrain_speed = {}
    all_results_raw = []
    traces_by_case: Dict[str, List[Dict[str, np.ndarray]]] = {"mpc_only": [], "mpc_rl": []}
    sim_dt_captured = 0.01

    while results_collected < expected_results:
        try:
            res = result_queue.get(timeout=1.0)
        except queue.Empty:
            if not any(w.is_alive() for w in workers):
                print("[compare] All workers have exited unexpectedly.")
                break
            continue

        if res["type"] == "result":
            ep_metrics = res["metrics"]
            ep_traces = res["traces"]
            case_name = res["case"]
            terrain = res["terrain"]
            command_forward = float(res.get("command_forward", getattr(ep_metrics, "command_forward", cfg_eval.command_forward)))

            all_rows.append(ep_metrics)
            traces_by_case.setdefault(case_name, []).append(ep_traces)
            terrain_rows_by_terrain_speed.setdefault((terrain, command_forward), []).append(ep_metrics)
            all_results_raw.append(res)
            sim_dt_captured = res["sim_dt"]
            results_collected += 1
        elif res["type"] == "error":
            case_errors.append(res["message"])
            results_collected += 1

    for w in workers:
        w.join(timeout=5.0)
        if w.is_alive():
            w.terminate()

    if not all_rows:
        raise RuntimeError("No completed episodes were recorded in any case.")

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "terrain",
                "command_forward",
                "stair_step_height",
                "case",
                "episode_idx",
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
                "total_steps",
            ]
        )
        for r in all_rows:
            w.writerow(
                [
                    r.terrain,
                    r.command_forward,
                    r.stair_step_height,
                    r.case,
                    r.episode_idx,
                    r.course_success,
                    r.course_complete,
                    r.goal_crossed,
                    r.completion_time_s,
                    r.strict_completion_time_s,
                    r.mean_tracking_error,
                    r.mean_cmd_speed_nominal,
                    r.mean_actual_speed_nominal,
                    r.mean_swing_time_s,
                    r.mean_cot,
                    r.total_steps,
                ]
            )

    def _row_matches(
        row: EpisodeMetrics,
        terrain: str,
        case: str,
        command_forward: float | None = None,
        stair_step_height: float | None = None,
    ) -> bool:
        if row.terrain != terrain or row.case != case:
            return False
        if command_forward is not None and abs(float(row.command_forward) - float(command_forward)) >= 1e-9:
            return False
        if stair_step_height is not None and abs(float(row.stair_step_height) - float(stair_step_height)) >= 1e-9:
            return False
        return True

    def _summary(
        terrain: str,
        command_forward: float,
        case: str,
        stair_step_height: float | None = None,
    ) -> Tuple[float, float, float, float, float, float]:
        rows = [
            r
            for r in all_rows
            if _row_matches(r, terrain, case, command_forward=command_forward, stair_step_height=stair_step_height)
        ]
        success_rate = float(np.mean([r.course_success for r in rows])) if rows else float("nan")
        complete_rate = float(np.mean([r.course_complete for r in rows])) if rows else float("nan")
        mean_time = _safe_nanmean([r.completion_time_s for r in rows]) if rows else float("nan")
        mean_time_strict = _safe_nanmean([r.strict_completion_time_s for r in rows]) if rows else float("nan")
        mean_track = _safe_nanmean([r.mean_tracking_error for r in rows]) if rows else float("nan")
        mean_cot = _safe_nanmean([r.mean_cot for r in rows]) if rows else float("nan")
        return success_rate, complete_rate, mean_time, mean_time_strict, mean_track, mean_cot

    def _mean_metric(
        terrain: str,
        command_forward: float | None,
        case: str,
        attr: str,
        stair_step_height: float | None = None,
    ) -> float:
        rows = [
            r
            for r in all_rows
            if _row_matches(r, terrain, case, command_forward=command_forward, stair_step_height=stair_step_height)
        ]
        values = [getattr(r, attr) for r in rows]
        return _safe_nanmean(values) if values else float("nan")

    def _mean_step_height_residual(
        terrain: str,
        command_forward: float | None,
        case: str,
        stair_step_height: float | None = None,
    ) -> float:
        values: List[float] = []
        for task_res in all_results_raw:
            if task_res["terrain"] != terrain or task_res["case"] != case:
                continue
            if command_forward is not None and abs(float(task_res.get("command_forward", command_forward)) - float(command_forward)) >= 1e-9:
                continue
            if stair_step_height is not None and abs(float(task_res.get("stair_step_height", stair_step_height)) - float(stair_step_height)) >= 1e-9:
                continue
            trace = task_res.get("traces", {})
            residual = np.asarray(trace.get("step_height_residual", np.zeros(0, dtype=np.float32)), dtype=np.float32)
            if residual.size > 0:
                values.append(float(np.mean(residual)))
        return _safe_nanmean(values) if values else float("nan")

    def _mean_trace(
        terrain: str,
        command_forward: float | None,
        case: str,
        trace_key: str,
        stair_step_height: float | None = None,
    ) -> float:
        values: List[float] = []
        for task_res in all_results_raw:
            if task_res["terrain"] != terrain or task_res["case"] != case:
                continue
            if command_forward is not None and abs(float(task_res.get("command_forward", command_forward)) - float(command_forward)) >= 1e-9:
                continue
            if stair_step_height is not None and abs(float(task_res.get("stair_step_height", stair_step_height)) - float(stair_step_height)) >= 1e-9:
                continue
            trace = task_res.get("traces", {})
            arr = np.asarray(trace.get(trace_key, np.zeros(0, dtype=np.float32)), dtype=np.float32)
            finite = arr[np.isfinite(arr)]
            if finite.size > 0:
                values.append(float(np.mean(finite)))
        return _safe_nanmean(values) if values else float("nan")

    def _mean_base_x(
        terrain: str,
        command_forward: float | None,
        case: str,
        stair_step_height: float | None = None,
    ) -> float:
        values: List[float] = []
        for task_res in all_results_raw:
            if task_res["terrain"] != terrain or task_res["case"] != case:
                continue
            if command_forward is not None and abs(float(task_res.get("command_forward", command_forward)) - float(command_forward)) >= 1e-9:
                continue
            if stair_step_height is not None and abs(float(task_res.get("stair_step_height", stair_step_height)) - float(stair_step_height)) >= 1e-9:
                continue
            trace = task_res.get("traces", {})
            base_x = np.asarray(trace.get("base_x", np.zeros(0, dtype=np.float32)), dtype=np.float32)
            if base_x.size > 0:
                values.append(float(np.mean(base_x)))
        return _safe_nanmean(values) if values else float("nan")

    print("\n[compare] Summary")
    for terrain in terrain_order:
        print(f"  Terrain={terrain}")
        for command_forward in speed_values:
            print(f"    Command forward={command_forward:.3f} m/s")
            if args.mode in ("both", "mpc_only"):
                mpc_success, mpc_complete, mpc_time, mpc_time_strict, mpc_track, mpc_cot = _summary(
                    terrain, command_forward, "mpc_only"
                )
                print(
                    f"      MPC-only : success_rate={mpc_success:.3f}, complete_rate={mpc_complete:.3f}, "
                    f"completion_time_s={mpc_time:.3f}, strict_completion_time_s={mpc_time_strict:.3f}, mean_tracking_error={mpc_track:.4f}, mean_cot={mpc_cot:.4f}"
                )
            if args.mode in ("both", "mpc_rl"):
                rl_success, rl_complete, rl_time, rl_time_strict, rl_track, rl_cot = _summary(
                    terrain, command_forward, "mpc_rl"
                )
                print(
                    f"      MPC+RL   : success_rate={rl_success:.3f}, complete_rate={rl_complete:.3f}, "
                    f"completion_time_s={rl_time:.3f}, strict_completion_time_s={rl_time_strict:.3f}, mean_tracking_error={rl_track:.4f}, mean_cot={rl_cot:.4f}"
                )

            terrain_speed_rows = [
                r
                for r in all_rows
                if r.terrain == terrain and abs(float(r.command_forward) - float(command_forward)) < 1e-9
            ]
            if not terrain_speed_rows:
                continue

            traces_for_terrain = {"mpc_only": [], "mpc_rl": []}
            for task_res in all_results_raw:
                if task_res["terrain"] == terrain and abs(float(task_res.get("command_forward", command_forward)) - float(command_forward)) < 1e-9:
                    traces_for_terrain[task_res["case"]].append(task_res["traces"])

            recorded_cases = {r.case for r in terrain_speed_rows}
            if args.mode == "both" and {"mpc_only", "mpc_rl"}.issubset(recorded_cases):
                speed_label = _speed_tag(command_forward)
                out_png_tracking = os.path.join(
                    args.output_dir,
                    f"go2_mpc_vs_mpc_rl_{terrain}_speed_{speed_label}_tracking_{stamp}.png",
                )
                out_png_base_z = os.path.join(
                    args.output_dir,
                    f"go2_mpc_vs_mpc_rl_{terrain}_speed_{speed_label}_base_z_{stamp}.png",
                )
                _plot_velocity_tracking_by_speed(
                    out_png_tracking,
                    terrain,
                    command_forward,
                    traces_for_terrain,
                    sim_dt=sim_dt_captured,
                )
                _plot_base_z_by_speed(
                    out_png_base_z,
                    terrain,
                    command_forward,
                    traces_for_terrain,
                    sim_dt=sim_dt_captured,
                )

    # Per-speed summary plots requested by user.
    for terrain in terrain_order:
        case_labels: Dict[str, str] = {}
        if args.mode in ("both", "mpc_only"):
            case_labels["mpc_only"] = "MPC-only"
        if args.mode in ("both", "mpc_rl"):
            case_labels["mpc_rl"] = "MPC+RL"

        tracking_series: Dict[str, List[float]] = {}
        success_series: Dict[str, List[float]] = {}
        completion_series: Dict[str, List[float]] = {}
        success_rate_series: Dict[str, List[float]] = {}
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
            tracking_series[label] = [
                _mean_metric(terrain, speed, case_key, "mean_tracking_error") for speed in speed_values
            ]
            success_series[label] = [
                _mean_metric(terrain, speed, case_key, "course_success") for speed in speed_values
            ]
            completion_series[label] = [
                _mean_metric(terrain, speed, case_key, "completion_time_s") for speed in speed_values
            ]
            success_rate_series[label] = [
                _mean_metric(terrain, speed, case_key, "course_success") for speed in speed_values
            ]
            residual_series[label] = [
                _mean_step_height_residual(terrain, speed, case_key) for speed in speed_values
            ]
            roll_series[label] = [_mean_trace(terrain, speed, case_key, "roll_error_rad") for speed in speed_values]
            pitch_series[label] = [_mean_trace(terrain, speed, case_key, "pitch_error_rad") for speed in speed_values]
            yaw_series[label] = [_mean_trace(terrain, speed, case_key, "yaw_error_rad") for speed in speed_values]
            torque_series[label] = [_mean_trace(terrain, speed, case_key, "joint_torque_mean_abs") for speed in speed_values]
            base_z_series[label] = [_mean_trace(terrain, speed, case_key, "base_z") for speed in speed_values]
            cot_series[label] = [
                _mean_metric(terrain, speed, case_key, "mean_cot") for speed in speed_values
            ]
            swing_foot_z_series[label] = [_mean_trace(terrain, speed, case_key, "swing_foot_z") for speed in speed_values]
            if case_key == "mpc_rl":
                swing_series[label] = [
                    _mean_metric(terrain, speed, case_key, "mean_swing_time_s") for speed in speed_values
                ]
            if terrain == "procedural":
                base_x_series[label] = [_mean_base_x(terrain, speed, case_key) for speed in speed_values]

        out_tracking = os.path.join(
            args.output_dir,
            f"go2_mpc_vs_mpc_rl_{terrain}_tracking_error_vs_speed_{stamp}.png",
        )
        _plot_metric_by_speed(
            out_tracking,
            terrain,
            speed_values,
            tracking_series,
            ylabel="Mean tracking error [m/s]",
            title="Mean speed tracking error vs speed",
        )

        out_completion = os.path.join(
            args.output_dir,
            f"go2_mpc_vs_mpc_rl_{terrain}_completion_time_vs_speed_{stamp}.png",
        )
        _plot_metric_by_speed(
            out_completion,
            terrain,
            speed_values,
            completion_series,
            ylabel="Mean completion time [s]",
            title="Mean completion time vs speed",
        )

        if terrain != "stairs":
            out_success = os.path.join(
                args.output_dir,
                f"go2_mpc_vs_mpc_rl_{terrain}_success_counts_vs_speed_{stamp}.png",
            )
            # success_series currently holds mean success rate per speed; convert to counts
            success_counts_series: Dict[str, List[int]] = {}
            for label, rates in success_series.items():
                counts: List[int] = []
                for rate, speed in zip(rates, speed_values):
                    # derive total episodes per speed/case from recorded rows
                    rows = [
                        r
                        for r in all_rows
                        if r.terrain == terrain and abs(float(r.command_forward) - float(speed)) < 1e-9 and r.case == ("mpc_only" if label == "MPC-only" else "mpc_rl")
                    ]
                    total = len(rows)
                    if np.isnan(rate):
                        counts.append(0)
                    else:
                        counts.append(int(round(float(rate) * total)))
                success_counts_series[label] = counts

            _plot_success_counts_by_speed(
                out_success,
                terrain,
                speed_values,
                success_counts_series,
                title="Successful episodes by speed and mode",
            )

            out_success_rate = os.path.join(
                args.output_dir,
                f"go2_mpc_vs_mpc_rl_{terrain}_success_rate_vs_speed_{stamp}.png",
            )
            _plot_metric_by_speed(
                out_success_rate,
                terrain,
                speed_values,
                success_rate_series,
                ylabel="Success rate",
                title="Success rate vs speed",
            )

        out_residual = os.path.join(
            args.output_dir,
            f"go2_mpc_vs_mpc_rl_{terrain}_step_height_residual_vs_speed_{stamp}.png",
        )
        _plot_metric_by_speed(
            out_residual,
            terrain,
            speed_values,
            residual_series,
            ylabel="Mean step height residual [m]",
            title="Mean step height residual vs speed",
        )

        out_attitude = os.path.join(
            args.output_dir,
            f"go2_mpc_vs_mpc_rl_{terrain}_attitude_error_vs_speed_{stamp}.png",
        )
        _plot_attitude_stability_by_speed(
            out_attitude,
            terrain,
            speed_values,
            roll_series,
            pitch_series,
            yaw_series,
        )

        out_torque = os.path.join(
            args.output_dir,
            f"go2_mpc_vs_mpc_rl_{terrain}_mean_joint_torque_vs_speed_{stamp}.png",
        )
        _plot_metric_by_speed(
            out_torque,
            terrain,
            speed_values,
            torque_series,
            ylabel="Mean joint torque |tau| [Nm]",
            title="Mean joint torque magnitude vs speed",
        )

        out_cot = os.path.join(
            args.output_dir,
            f"go2_mpc_vs_mpc_rl_{terrain}_cot_vs_speed_{stamp}.png",
        )
        _plot_metric_by_speed(
            out_cot,
            terrain,
            speed_values,
            cot_series,
            ylabel="Cost of Transport",
            title="Cost of Transport (CoT) vs speed",
        )

        out_base_z_speed = os.path.join(
            args.output_dir,
            f"go2_mpc_vs_mpc_rl_{terrain}_base_z_vs_speed_{stamp}.png",
        )
        _plot_metric_by_speed(
            out_base_z_speed,
            terrain,
            speed_values,
            base_z_series,
            ylabel="Mean base z [m]",
            title="Mean base z vs speed",
        )

        out_swing_foot_z_speed = os.path.join(
            args.output_dir,
            f"go2_mpc_vs_mpc_rl_{terrain}_swing_foot_z_vs_speed_{stamp}.png",
        )
        _plot_metric_by_speed(
            out_swing_foot_z_speed,
            terrain,
            speed_values,
            swing_foot_z_series,
            ylabel="Mean swing foot z [m]",
            title="Mean swing foot z vs speed",
        )

        if terrain == "procedural" and base_x_series:
            out_base_x = os.path.join(
                args.output_dir,
                f"go2_mpc_vs_mpc_rl_{terrain}_base_x_vs_speed_{stamp}.png",
            )
            _plot_metric_by_speed(
                out_base_x,
                terrain,
                speed_values,
                base_x_series,
                ylabel="Mean base x reached [m]",
                title="Mean base x reached vs speed",
            )

        if swing_series:
            out_swing = os.path.join(
                args.output_dir,
                f"go2_mpc_vs_mpc_rl_{terrain}_swing_time_mpc_rl_vs_speed_{stamp}.png",
            )
            _plot_metric_by_speed(
                out_swing,
                terrain,
                speed_values,
                swing_series,
                ylabel="Mean swing time [s]",
                title="Mean swing time (MPC+RL) vs speed",
            )

        if terrain == "stairs":
            speed_success_rate_by_mode: Dict[str, Dict[str, List[float]]] = {}
            height_success_rate_by_mode: Dict[str, Dict[str, List[float]]] = {}
            speed_success_counts_by_mode: Dict[str, Dict[str, List[float]]] = {}
            height_success_counts_by_mode: Dict[str, Dict[str, List[float]]] = {}

            for case_key, label in case_labels.items():
                speed_success_rate_by_mode[label] = {}
                height_success_rate_by_mode[label] = {}
                speed_success_counts_by_mode[label] = {}
                height_success_counts_by_mode[label] = {}

                for height in stair_height_values:
                    height_label = f"{height:.2f} m"
                    rate_values = [
                        _mean_metric(terrain, speed, case_key, "course_success", stair_step_height=height)
                        for speed in speed_values
                    ]
                    count_values = []
                    for speed in speed_values:
                        rows = [
                            r
                            for r in all_rows
                            if r.terrain == terrain
                            and r.case == case_key
                            and abs(float(r.command_forward) - float(speed)) < 1e-9
                            and abs(float(r.stair_step_height) - float(height)) < 1e-9
                        ]
                        count_values.append(int(sum(1 for r in rows if float(r.course_success) > 0.5)))
                    speed_success_rate_by_mode[label][height_label] = rate_values
                    speed_success_counts_by_mode[label][height_label] = count_values

                for speed in speed_values:
                    speed_label = f"{speed:.2f} m/s"
                    rate_values = [
                        _mean_metric(terrain, speed, case_key, "course_success", stair_step_height=height)
                        for height in stair_height_values
                    ]
                    count_values = []
                    for height in stair_height_values:
                        rows = [
                            r
                            for r in all_rows
                            if r.terrain == terrain
                            and r.case == case_key
                            and abs(float(r.command_forward) - float(speed)) < 1e-9
                            and abs(float(r.stair_step_height) - float(height)) < 1e-9
                        ]
                        count_values.append(int(sum(1 for r in rows if float(r.course_success) > 0.5)))
                    height_success_rate_by_mode[label][speed_label] = rate_values
                    height_success_counts_by_mode[label][speed_label] = count_values

            out_success_summary = os.path.join(
                args.output_dir,
                f"go2_mpc_vs_mpc_rl_{terrain}_success_summary_speed_height_{stamp}.png",
            )
            _plot_success_summary_by_speed_and_height(
                out_success_summary,
                terrain,
                speed_values,
                stair_height_values,
                speed_success_rate_by_mode,
                height_success_rate_by_mode,
                ylabel="Success rate",
                title="Success rate by speed and stair height",
                value_format="{:.2f}",
            )

            out_success_counts = os.path.join(
                args.output_dir,
                f"go2_mpc_vs_mpc_rl_{terrain}_successful_episodes_speed_height_{stamp}.png",
            )
            _plot_success_summary_by_speed_and_height(
                out_success_counts,
                terrain,
                speed_values,
                stair_height_values,
                speed_success_counts_by_mode,
                height_success_counts_by_mode,
                ylabel="Successful episodes",
                title="Successful episodes by speed, mode, and stair height",
                value_format="{:.0f}",
            )

    recorded_cases = {r.case for r in all_rows}
    if args.mode == "both" and {"mpc_only", "mpc_rl"}.issubset(recorded_cases):
        out_png = os.path.join(args.output_dir, f"go2_mpc_vs_mpc_rl_plots_{stamp}.png")
        out_png_profiles = os.path.join(args.output_dir, f"go2_mpc_vs_mpc_rl_speed_stepheight_{stamp}.png")
        _plot_results(out_png, all_rows, traces_by_case=traces_by_case, sim_dt=float(sim_dt_captured))
        _plot_speed_and_step_height_profiles(out_png_profiles, traces_by_case, sim_dt=float(sim_dt_captured))

    print(f"[compare] Saved CSV : {out_csv}")
    if args.mode == "both" and {"mpc_only", "mpc_rl"}.issubset(recorded_cases):
        print("[compare] Plots saved in output-dir")
    elif args.mode == "both":
        print("[compare] Plot skipped: both controller cases were not completed on at least one terrain/speed combination.")

    if case_errors:
        print("[compare] Case errors:")
        for m in case_errors:
            print(f"  - {m}")


if __name__ == "__main__":
    main()

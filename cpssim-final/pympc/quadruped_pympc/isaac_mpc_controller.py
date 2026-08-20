"""
Direct MPC controller for Isaac Sim integration (no ROS).
Provides synchronous MPC computation without message passing overhead.
"""

import numpy as np
import time
from scipy.spatial.transform import Rotation
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from quadruped_pympc import config as cfg
from quadruped_pympc.interfaces.srbd_controller_interface import SRBDControllerInterface
from quadruped_pympc.interfaces.wb_interface import WBInterface
from quadruped_pympc.helpers.legs_attr import LegsAttr
import os
import importlib.util
import inspect

# HeightMap adapter: prefer project location, fallback to legacy file location
try:
    from pympc.isaac.heightmap_isaac import HeightMapIsaac
except Exception:
    # Attempt to load from IsaacLabFromTemu/src/dog/heightmap_isaac.py by path
    try:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        candidate = os.path.join(base_dir, 'IsaacLabFromTemu', 'src', 'dog', 'heightmap_isaac.py')
        if os.path.exists(candidate):
            spec = importlib.util.spec_from_file_location('dog.heightmap_isaac', candidate)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            HeightMapIsaac = getattr(module, 'HeightMapIsaac')
            try:
                print(f"[IsaacMPCController] HeightMapIsaac loaded from file: {candidate}")
            except Exception:
                pass
        else:
            HeightMapIsaac = None
    except Exception:
        HeightMapIsaac = None

# Print resolved path if available
try:
    if HeightMapIsaac is not None:
        try:
            hm_path = inspect.getfile(HeightMapIsaac)
            print(f"[IsaacMPCController] Resolved HeightMapIsaac path: {hm_path}")
        except Exception:
            print("[IsaacMPCController] HeightMapIsaac loaded but path could not be resolved")
    else:
        print("[IsaacMPCController] HeightMapIsaac not available")
except Exception:
    pass



class IsaacMPCController:
    """
    Standalone MPC controller for Isaac Sim.
    Computes control synchronously without ROS overhead.
    """
    
    def __init__(self, raycast_fn=None):
        """Initialize MPC controller and interfaces."""
        self.legs_order = ["FL", "FR", "RL", "RR"]
        
        # Initialize MPC interfaces
        # Create initial feet positions as LegsAttr (not numpy array)
        zero_feet = LegsAttr(
            FL=np.zeros(3), FR=np.zeros(3),
            RL=np.zeros(3), RR=np.zeros(3)
        )
        self.wb_interface = WBInterface(initial_feet_pos=zero_feet, legs_order=self.legs_order)
        self.srbd_controller_interface = SRBDControllerInterface()
        self._nominal_step_height = float(self.wb_interface.step_height)
        self._rl_step_height_enabled = bool(cfg.simulation_params.get('rl_step_height_enabled', False))
        self._rl_step_height_residual_m = 0.0
        self._rl_residual_max_abs_m = float(cfg.simulation_params.get('rl_step_height_residual_max_abs', 0.08))
        self._rl_step_height_min_m = float(cfg.simulation_params.get('rl_step_height_min', 0.08 * float(cfg.hip_height)))
        self._rl_step_height_max_m = float(cfg.simulation_params.get('rl_step_height_max', 1.15 * float(cfg.hip_height)))
        self._rl_step_height_smoothing = float(cfg.simulation_params.get('rl_step_height_residual_smoothing', 0.25))
        self._last_rl_snapshot = None
        # Env-provided context appended to RL observation vector.
        self._rl_obs_residual_gate = 0.0
        self._rl_obs_swing_time = float(cfg.simulation_params.get('adaptive_swing_time', 0.20))
        
        # State tracking
        self.last_mpc_time = time.time()
        self.mpc_freq = 100.0  # Hz
        
        # Joystick/reference state
        self.ref_base_lin_vel = np.array([0.0, 0.0, 0.0])
        self.ref_base_ang_vel = np.array([0.0, 0.0, 0.0])
        self.ref_base_pitch_offset = 0.0
        
        # MPC outputs
        self.nmpc_GRFs = LegsAttr(FL=np.zeros(3), FR=np.zeros(3), RL=np.zeros(3), RR=np.zeros(3))
        self.nmpc_footholds = LegsAttr(FL=np.zeros(3), FR=np.zeros(3), RL=np.zeros(3), RR=np.zeros(3))
        self.nmpc_joints_pos = LegsAttr(FL=np.zeros(3), FR=np.zeros(3), RL=np.zeros(3), RR=np.zeros(3))
        self.nmpc_joints_vel = LegsAttr(FL=np.zeros(3), FR=np.zeros(3), RL=np.zeros(3), RR=np.zeros(3))  
        self.nmpc_joints_acc = LegsAttr(FL=np.zeros(3), FR=np.zeros(3), RL=np.zeros(3), RR=np.zeros(3))  #defined in the wb_interface but not used
        self.nmpc_predicted_state = np.zeros(12)        #defined in the wb_interface but not used
        self.best_sample_freq = self.wb_interface.pgg.step_freq  #not needed if adaptive gait is enabled

        # MPC instrumentation / diagnostics
        self._instrumentation_enabled = bool(cfg.simulation_params.get('mpc_instrumentation_enabled', True))
        self._instrumentation_print_interval = int(max(1, cfg.simulation_params.get('mpc_instrumentation_print_interval', 25)))
        self._instrumentation_step_counter = 0
        self._last_solver_diag = {}
        self._last_mpc_compute_duration_s = 0.0
        self._last_mpc_solve_status = None
        self._acados_maxiter_count_total = 0
        self._acados_maxiter_count_episode = 0
        self._acados_minstep_count_total = 0
        self._acados_minstep_count_episode = 0
        self._acados_failure_count_total = 0
        self._acados_failure_count_episode = 0
        
        # Torque limits (per-joint: [hip, thigh, calf])
        self.tau_max = np.array([23.7, 23.7, 45.43]) * 0.9  # Soft limit scalar
        self.tau = LegsAttr(
            FL=np.zeros(3), FR=np.zeros(3), 
            RL=np.zeros(3), RR=np.zeros(3)
        )
        
        # Impedance PD gains (joint-space overlay on top of feedforward torques)
        # These match the TrajectoryGeneratorMsg kp/kd from the ROS controller path
        self.impedance_kp = np.ones(3) * cfg.simulation_params['impedence_joint_position_gain']  # 10.0
        self.impedance_kd = np.ones(3) * cfg.simulation_params['impedence_joint_velocity_gain']  # 2.0
        
        # Joint index mapping in qpos/qvel
        # qpos: [pos(3), quat(4), joints(12)] -> joints start at index 7
        # qvel: [lin_vel(3), ang_vel(3), joints(12)] -> joints start at index 6
        self.legs_qpos_idx = LegsAttr(
            FL=np.array([7, 8, 9]), FR=np.array([10, 11, 12]),
            RL=np.array([13, 14, 15]), RR=np.array([16, 17, 18])
        )
        self.legs_qvel_idx = LegsAttr(
            FL=np.array([6, 7, 8]), FR=np.array([9, 10, 11]),
            RL=np.array([12, 13, 14]), RR=np.array([15, 16, 17])
        )
        
        print("[IsaacMPCController] Initialized (no ROS)")

        # Initialize heightmap adapters (one per leg). Prefer an externally
        # provided `raycast_fn` (useful when running inside Isaac/DogRunner).
        # Otherwise fall back to the simple flat-ground raycast.
        try:
            if HeightMapIsaac is None:
                raise ImportError("HeightMapIsaac adapter not available")

            # choose provided raycast if available (no simple fallback)
            chosen_raycast = raycast_fn

            rows = int(cfg.simulation_params.get('heightmap_num_rows', 10))
            cols = int(cfg.simulation_params.get('heightmap_num_cols', 7))
            res = float(cfg.simulation_params.get('heightmap_resolution', 0.06))
            max_dist = float(cfg.simulation_params.get('heightmap_raycast_max_dist', 1.0))
            height_bias = float(cfg.simulation_params.get('heightmap_height_bias', 0.0))

            self.heightmaps = LegsAttr(
                FL=HeightMapIsaac(num_rows=rows, num_cols=cols, dist_x=res, dist_y=res, raycast_fn=chosen_raycast, raycast_max_dist=max_dist, height_bias=height_bias),
                FR=HeightMapIsaac(num_rows=rows, num_cols=cols, dist_x=res, dist_y=res, raycast_fn=chosen_raycast, raycast_max_dist=max_dist, height_bias=height_bias),
                RL=HeightMapIsaac(num_rows=rows, num_cols=cols, dist_x=res, dist_y=res, raycast_fn=chosen_raycast, raycast_max_dist=max_dist, height_bias=height_bias),
                RR=HeightMapIsaac(num_rows=rows, num_cols=cols, dist_x=res, dist_y=res, raycast_fn=chosen_raycast, raycast_max_dist=max_dist, height_bias=height_bias),
            )
            print("[IsaacMPCController] HeightMapIsaac adapters initialized.")
        except Exception as e:
            print(f"[IsaacMPCController] HeightMap adapters not initialized: {e}")
            self.heightmaps = None

    def enable_rl_step_height(self, enabled: bool) -> None:
        self._rl_step_height_enabled = bool(enabled)

    def set_step_height_residual(self, residual_m: float) -> None:
        """Set RL residual action in meters (applied as additive delta to nominal step height)."""
        target = float(np.clip(residual_m, -self._rl_residual_max_abs_m, self._rl_residual_max_abs_m))
        alpha = float(np.clip(self._rl_step_height_smoothing, 0.0, 1.0))
        self._rl_step_height_residual_m = (1.0 - alpha) * self._rl_step_height_residual_m + alpha * target

    def set_adaptive_swing_time(self, swing_time_s: float) -> None:
        self.wb_interface.set_adaptive_swing_time(float(swing_time_s))


    def _apply_step_height_residual(self) -> float:
        if self._rl_step_height_enabled:
            effective = self._nominal_step_height + self._rl_step_height_residual_m
        else:
            effective = self._nominal_step_height
        effective = float(np.clip(effective, self._rl_step_height_min_m, self._rl_step_height_max_m))
        self.wb_interface.step_height = effective
        return effective

    def get_rl_state_snapshot(self) -> Optional[dict]:
        # Return the latest rich controller state (dict) used by the env for
        # reward/termination/logging (raw values, not the compact RL obs).
        return self._last_rl_snapshot

    def set_rl_observation_context(
        self,
        residual_gate: float,
        swing_time: Optional[float] = None,
    ) -> None:
        """Set env-side context features appended to RL observation vector."""
        self._rl_obs_residual_gate = float(np.clip(residual_gate, 0.0, 1.0))
        if swing_time is not None:
            self._rl_obs_swing_time = float(swing_time)

    def reset_for_new_episode(self) -> None:
        """Reset episode-dependent MPC/solver state to avoid warm-start carryover."""
        self.last_mpc_time = time.time()
        self.nmpc_GRFs = LegsAttr(FL=np.zeros(3), FR=np.zeros(3), RL=np.zeros(3), RR=np.zeros(3))
        self.nmpc_footholds = LegsAttr(FL=np.zeros(3), FR=np.zeros(3), RL=np.zeros(3), RR=np.zeros(3))
        self.nmpc_joints_pos = LegsAttr(FL=np.zeros(3), FR=np.zeros(3), RL=np.zeros(3), RR=np.zeros(3))
        self.nmpc_joints_vel = LegsAttr(FL=np.zeros(3), FR=np.zeros(3), RL=np.zeros(3), RR=np.zeros(3))
        self.nmpc_joints_acc = LegsAttr(FL=np.zeros(3), FR=np.zeros(3), RL=np.zeros(3), RR=np.zeros(3))
        self.nmpc_predicted_state = np.zeros(12)
        self._last_solver_diag = {}
        self._last_mpc_compute_duration_s = 0.0
        self._last_mpc_solve_status = None
        self._acados_maxiter_count_episode = 0
        self._acados_minstep_count_episode = 0
        self._acados_failure_count_episode = 0
        self._rl_obs_residual_gate = 0.0
        self._rl_obs_swing_time = float(cfg.simulation_params.get('adaptive_swing_time', 0.20))

        try:
            self.srbd_controller_interface.reset()
        except Exception:
            pass

        try:
            self.wb_interface.reset_for_new_episode()
        except Exception:
            pass

    def get_rl_observation_vector(self) -> np.ndarray:
        """Return a compact observation vector for RL (fixed size, float32)."""
        # Includes measured foot-force channels in addition to planned contact.
        obs = np.zeros(86, dtype=np.float32)
        snap = self._last_rl_snapshot
        if snap is None:
            obs[84] = float(self._rl_obs_residual_gate)
            obs[85] = float(getattr(self, '_rl_obs_swing_time', 0.20))
            return obs

        ref_lin = np.asarray(snap.get('ref_base_lin_vel', np.zeros(3)), dtype=np.float32)
        ref_ang = np.asarray(snap.get('ref_base_ang_vel', np.zeros(3)), dtype=np.float32)
        pos = np.asarray(snap.get('position', np.zeros(3)), dtype=np.float32)
        lin = np.asarray(snap.get('linear_velocity', np.zeros(3)), dtype=np.float32)
        ang = np.asarray(snap.get('base_ang_vel_B', np.zeros(3)), dtype=np.float32)
        euler = np.asarray(snap.get('base_ori_euler_xyz', np.zeros(3)), dtype=np.float32)
        contact = np.asarray(snap.get('current_contact', np.zeros(4)), dtype=np.float32)
        measured_contact_forces = np.asarray(snap.get('measured_contact_forces', np.zeros(4)), dtype=np.float32)

        obs[0:3] = ref_lin
        obs[3:6] = lin
        obs[6:9] = ang
        obs[9:12] = euler[0:3]
        obs[12] = float(snap.get('step_height_effective', self._nominal_step_height))
        obs[13] = float(self._rl_step_height_residual_m)
        obs[14:18] = contact
        # Squash measured forces to [0, 1) with tanh to keep scale bounded.
        obs[18:22] = np.tanh(np.maximum(0.0, measured_contact_forces) / 20.0)

        # CHANGED: Base position context. Mask out absolute World X and Y to prevent spatial overfitting.
        obs[22] = 0.0       # Was pos[0]
        obs[23] = 0.0       # Was pos[1]
        obs[24] = pos[2]    # Keep absolute base height

        # Command-tracking error features.
        obs[25] = float(ref_lin[0] - lin[0])
        obs[26] = float(ref_lin[1] - lin[1])
        obs[27] = float(ref_ang[2] - ang[2])

        # CHANGED: Sparse Sampling for Heightmaps
        idx = 28
        
        # Define the 5-point cross pattern around the foot (in meters)
        # Assuming dist_x/y is ~0.06m, a 0.1m offset looks roughly 1.5 grid cells away
        sparse_offsets = [
            (0.0, 0.0),    # Center (directly under foot)
            (0.12, 0.0),   # Front
            (-0.12, 0.0),  # Back
            (0.0, 0.06),   # Left
            (0.0, -0.06)   # Right
        ]
        
        feet_pos_flat = np.asarray(snap.get('feet_pos_w', np.zeros(12)), dtype=np.float32)

        for i, leg in enumerate(self.legs_order):
            hm = None if self.heightmaps is None else self.heightmaps[leg]
            
            # Extract the 3D position for this specific leg from the flattened array
            foot_pos = feet_pos_flat[i*3 : (i+1)*3]
            fallback_z = float(foot_pos[2])
            
            if hm is not None and hm.data is not None:
                # Get the 5 absolute heights
                absolute_heights = hm.get_sparse_samples([foot_pos[0], foot_pos[1]], sparse_offsets, fallback_z)
                
                # Convert to relative heights (relative to the robot's base Z)
                # If absolute_height > base Z, value is positive (step up)
                # If absolute_height < base Z, value is negative (drop off)
                for i in range(5):
                    obs[idx + i] = float(absolute_heights[i]) - float(pos[2])
            else:
                # If no heightmap data, pass relative height of 0.0 (assume flat ground relative to current foot pos)
                for i in range(5):
                    obs[idx + i] = fallback_z - float(pos[2])
                    
            idx += 5

        # Joint state augmentation for improved proprioception.
        joint_pos = np.asarray(snap.get('joint_positions', np.zeros(12)), dtype=np.float32)
        joint_vel = np.asarray(snap.get('joint_velocities', np.zeros(12)), dtype=np.float32)
        obs[48:60] = joint_pos[:12]
        obs[60:72] = joint_vel[:12]

        # Foothold tracking error in yaw-aligned body frame: planned - current.
        foothold_error_body = np.asarray(snap.get('foothold_error_body', np.zeros(12)), dtype=np.float32)
        obs[72:84] = foothold_error_body[:12]
        obs[84] = float(self._rl_obs_residual_gate)
        obs[85] = float(getattr(self, '_rl_obs_swing_time', 0.20))
        
        return obs
        
    def compute_control(
        self,
        position: np.ndarray,
        orientation_quat_xyzw: np.ndarray,
        linear_velocity: np.ndarray,
        angular_velocity: np.ndarray,
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray,
        feet_positions: np.ndarray,  # 12-dim flattened [FL, FR, RL, RR]
        hip_positions: np.ndarray,   # 12-dim flattened
        contact_forces: np.ndarray,  # 4-dim [FL, FR, RL, RR]
        mass_matrix: np.ndarray,     # (12, 12)
        bias_forces: np.ndarray,     # 12-dim
        feet_jacobians: np.ndarray,  # (4, 3, 18)
        feet_jacobian_dots: np.ndarray,  # (4, 3, 18) - time derivative of Jacobians
        feet_velocities: np.ndarray,     # 12-dim flattened [FL, FR, RL, RR]
        inertia: np.ndarray,         # (3, 3) or flat
        simulation_dt: float,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Compute control torques synchronously.
        
        Args:
            position: Base position (3D)
            orientation_quat_xyzw: Base orientation quaternion in xyzw format
            linear_velocity: Base linear velocity (3D, world frame)
            angular_velocity: Base angular velocity (3D, body frame)
            joint_positions: Joint positions (12D)
            joint_velocities: Joint velocities (12D)
            feet_positions: Feet positions (12D)
            hip_positions: Hip positions (12D)
            contact_forces: Per-foot contact state (4D: [FL, FR, RL, RR])
            mass_matrix: Leg mass matrices (12, 12)
            bias_forces: Coriolis/gravity bias (12D)
            feet_jacobians: Feet Jacobians (4, 3, 18)
            feet_jacobian_dots: Time derivative of feet Jacobians (4, 3, 18)
            feet_velocities: Feet velocities in world frame (12D)
            inertia: Base inertia (3, 3) or (9,)
            simulation_dt: Simulation timestep
            
        Returns:
            torques: Joint torques (12D)
            debug_info: Dictionary with debug info
        """
        
        # --- Convert quaternion and compute rotation ---
        try:
            R = Rotation.from_quat(orientation_quat_xyzw).as_matrix()
            base_ori_euler_xyz = Rotation.from_quat(orientation_quat_xyzw).as_euler('xyz')
        except Exception as e:
            print(f"[IsaacMPCController] Invalid quaternion {orientation_quat_xyzw}, using identity: {e}")
            R = np.eye(3)
            base_ori_euler_xyz = np.zeros(3)
            orientation_quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0])
        
        # --- Convert angular velocity to body frame ---
        base_ang_vel_B = R.T @ angular_velocity
        
        # --- Construct state vectors ---
        qpos = np.concatenate([position, orientation_quat_xyzw, joint_positions])
        qvel = np.concatenate([linear_velocity, base_ang_vel_B, joint_velocities])
        
        # --- Reshape feet/hip positions ---
        feet_pos_dict = LegsAttr(
            FL=feet_positions[0:3],
            FR=feet_positions[3:6],
            RL=feet_positions[6:9],
            RR=feet_positions[9:12]
        )
        
        hip_pos_dict = LegsAttr(
            FL=hip_positions[0:3],
            FR=hip_positions[3:6],
            RL=hip_positions[6:9],
            RR=hip_positions[9:12]
        )
        
        # --- Validate per-foot contact input ---
        # NMPC uses the planned contact sequence from WBInterface/PGG and does not
        # consume measured contact forces directly in this direct-Isaac path.
        # Keep this input strict to avoid ambiguous 12D-vs-4D handling.
        contact_forces = np.asarray(contact_forces).reshape(-1)
        if contact_forces.size != 4:
            raise ValueError(
                f"Expected contact_forces shape (4,) as [FL, FR, RL, RR], got {contact_forces.shape}"
            )

        step_height_effective = self._apply_step_height_residual()
        self.wb_interface.reference_pitch_offset = float(self.ref_base_pitch_offset)
        
        # --- Update whole-body interface state ---
        state_current, ref_state, contact_sequence, step_height, optimize_swing = \
            self.wb_interface.update_state_and_reference(
                com_pos=position,
                base_pos=position,
                base_lin_vel=linear_velocity,
                base_ori_euler_xyz=base_ori_euler_xyz,
                base_ang_vel=base_ang_vel_B,
                feet_pos=feet_pos_dict,
                hip_pos=hip_pos_dict,
                joints_pos=LegsAttr(
                    FL=joint_positions[0:3], FR=joint_positions[3:6],
                    RL=joint_positions[6:9], RR=joint_positions[9:12]
                ),
                heightmaps=self.heightmaps,
                legs_order=self.legs_order,
                simulation_dt=simulation_dt,
                ref_base_lin_vel=self.ref_base_lin_vel,
                ref_base_ang_vel=self.ref_base_ang_vel
            )

        # Visualization: create a debug image showing heightmap sample Z and adapted footholds
        try:
            if getattr(self, 'heightmaps', None) is not None and ref_state is not None:
                self._visualize_heightmaps(position, base_ori_euler_xyz[2], ref_state)
        except Exception:
            pass
        
        # --- Compute MPC only at desired frequency ---
        current_time = time.time()
        if current_time - self.last_mpc_time >= (1.0 / self.mpc_freq):
            mpc_compute_t0 = time.perf_counter()
            try:
                # NMPC expects inertia as flat 9-element array (row-major 3x3)
                if inertia.size == 9:
                    inertia_flat = inertia.flatten()
                elif inertia.size == 3:
                    # Diagonal only [Ixx, Iyy, Izz] -> full 3x3 -> flatten
                    inertia_flat = np.diag(inertia).flatten()
                else:
                    inertia_flat = np.zeros(9)
                    inertia_flat[:min(inertia.size, 9)] = inertia.flatten()[:min(inertia.size, 9)]
                
                res = self.srbd_controller_interface.compute_control(
                    state_current,
                    ref_state,
                    contact_sequence,
                    inertia_flat,
                    self.wb_interface.pgg.phase_signal,
                    self.wb_interface.pgg.step_freq,
                    optimize_swing
                )
                
                if isinstance(res, (tuple, list)) and len(res) == 7:
                    (self.nmpc_GRFs, self.nmpc_footholds, self.nmpc_joints_pos,
                     self.nmpc_joints_vel, self.nmpc_joints_acc, 
                     self.best_sample_freq, self.nmpc_predicted_state) = res
                
                # RTI feedback step if configured
                if cfg.mpc_params['type'] != 'sampling' and cfg.mpc_params.get('use_RTI', False):
                    self.srbd_controller_interface.compute_RTI()

                raw_status = getattr(self.srbd_controller_interface, 'last_solver_status', None)
                if raw_status is None:
                    raw_status = getattr(self.srbd_controller_interface.controller, 'previous_status', None)
                self._last_mpc_solve_status = raw_status
                self._last_solver_diag = self._collect_solver_diagnostics()
                self._last_mpc_compute_duration_s = float(time.perf_counter() - mpc_compute_t0)
                
                self.last_mpc_time = current_time
            except Exception as e:
                print(f"[IsaacMPCController] MPC computation error: {e}")
                import traceback
                traceback.print_exc()
                self._last_mpc_compute_duration_s = float(time.perf_counter() - mpc_compute_t0)
        
        # --- Compute stance and swing torques ---
        try:
            # Convert Jacobians to LegsAttr
            if not isinstance(feet_jacobians, LegsAttr):
                if feet_jacobians.ndim == 1:
                    feet_jacobians = feet_jacobians.reshape(4, 3, -1)
                feet_jac_dict = LegsAttr(
                    FL=feet_jacobians[0], FR=feet_jacobians[1],
                    RL=feet_jacobians[2], RR=feet_jacobians[3]
                )
            else:
                feet_jac_dict = feet_jacobians
            
            # Convert Jacobian dots to LegsAttr (actual data from Isaac dynamics)
            if not isinstance(feet_jacobian_dots, LegsAttr):
                if feet_jacobian_dots.ndim == 1:
                    feet_jacobian_dots = feet_jacobian_dots.reshape(4, 3, -1)
                feet_jac_dot_dict = LegsAttr(
                    FL=feet_jacobian_dots[0], FR=feet_jacobian_dots[1],
                    RL=feet_jacobian_dots[2], RR=feet_jacobian_dots[3]
                )
            else:
                feet_jac_dot_dict = feet_jacobian_dots
            
            # Feet velocities from Isaac Sim (actual, not placeholder)
            feet_vel_dict = LegsAttr(
                FL=feet_velocities[0:3], FR=feet_velocities[3:6],
                RL=feet_velocities[6:9], RR=feet_velocities[9:12]
            )
            
            # Passive forces (zero for ideal actuators in Isaac Sim)
            legs_qfrc_passive_dict = LegsAttr(
                FL=np.zeros(3), FR=np.zeros(3),
                RL=np.zeros(3), RR=np.zeros(3)
            )
            
            # Bias forces by leg (reshape from 12D)
            legs_qfrc_bias_dict = LegsAttr(
                FL=bias_forces[0:3], FR=bias_forces[3:6],
                RL=bias_forces[6:9], RR=bias_forces[9:12]
            )
            
            # Mass matrices by leg (block-diagonal extraction)
            legs_mass_matrix_dict = LegsAttr(
                FL=mass_matrix[0:3, 0:3], FR=mass_matrix[3:6, 3:6],
                RL=mass_matrix[6:9, 6:9], RR=mass_matrix[9:12, 9:12]
            )
            
            self.tau, pd_target_joints_pos, pd_target_joints_vel = \
                self.wb_interface.compute_stance_and_swing_torque(
                    simulation_dt,
                    qpos, qvel,
                    feet_jac_dict,
                    feet_jac_dot_dict,
                    feet_pos_dict,
                    feet_vel_dict,
                    legs_qfrc_passive_dict,
                    legs_qfrc_bias_dict,
                    legs_mass_matrix_dict,
                    self.nmpc_GRFs,
                    self.nmpc_footholds,
                    self.legs_qpos_idx,
                    self.legs_qvel_idx,
                    self.tau,
                    optimize_swing,
                    self.best_sample_freq,
                    self.nmpc_joints_pos,
                    self.nmpc_joints_vel,
                    self.nmpc_joints_acc,
                    self.nmpc_predicted_state,
                    mujoco_contact=contact_forces,
                )
            
            # Flatten tau per leg
            for leg in self.legs_order:
                self.tau[leg] = np.asarray(self.tau[leg]).flatten()
                
        except Exception as e:
            print(f"[IsaacMPCController] Swing/stance computation error: {e}")
            import traceback
            traceback.print_exc()
            self.tau = LegsAttr(FL=np.zeros(3), FR=np.zeros(3), RL=np.zeros(3), RR=np.zeros(3))
            pd_target_joints_pos = LegsAttr(FL=joint_positions[0:3], FR=joint_positions[3:6],
                                            RL=joint_positions[6:9], RR=joint_positions[9:12])
            pd_target_joints_vel = LegsAttr(FL=np.zeros(3), FR=np.zeros(3),
                                            RL=np.zeros(3), RR=np.zeros(3))
        
        # --- Add impedance PD overlay (joint-space compliance + damping) ---
        # This replicates the TrajectoryGeneratorMsg kp/kd that the ROS path applies
        # at the actuator level. Without this, there is zero joint damping and the
        # hip (abduction) joints overshoot at higher speeds.
        for i, leg in enumerate(self.legs_order):
            q_leg = joint_positions[i*3:(i+1)*3]
            qd_leg = joint_velocities[i*3:(i+1)*3]
            q_des = np.asarray(pd_target_joints_pos[leg]).flatten()
            qd_des = np.asarray(pd_target_joints_vel[leg]).flatten()
            tau_pd = self.impedance_kp * (q_des - q_leg) + self.impedance_kd * (qd_des - qd_leg)
            self.tau[leg] = self.tau[leg] + tau_pd
        
        # --- Apply torque limits ---
        for leg in self.legs_order:
            current_tau = np.asarray(self.tau[leg]).flatten()
            self.tau[leg] = np.clip(current_tau, -self.tau_max, self.tau_max)
        
        # --- Extract and return torques ---
        torques = np.concatenate([
            self.tau.FL.flatten(),
            self.tau.FR.flatten(),
            self.tau.RL.flatten(),
            self.tau.RR.flatten(),
        ])

        torque_abs = np.abs(torques)
        torque_l2 = float(np.linalg.norm(torques))
        torque_max_abs = float(np.max(torque_abs)) if torque_abs.size > 0 else 0.0
        torque_limit_max = float(np.max(self.tau_max)) if self.tau_max.size > 0 else 0.0
        # Compute per-joint saturation using per-joint limits tiled for each leg
        try:
            tau_limits_per_joint = np.tile(np.asarray(self.tau_max).flatten(), len(self.legs_order))
            torque_sat_ratio = float(np.mean(torque_abs >= (tau_limits_per_joint - 1e-6))) if torque_abs.size > 0 else 0.0
        except Exception:
            # Fallback to conservative scalar comparison if something unexpected occurs
            torque_sat_ratio = float(np.mean(torque_abs >= (self.tau_max.max() - 1e-6))) if torque_abs.size > 0 else 0.0
        mpc_diag = {
            "solve_status": -999 if self._last_mpc_solve_status is None else int(self._last_mpc_solve_status),
            "compute_duration_ms": float(self._last_mpc_compute_duration_s * 1e3),
            "torque_l2": torque_l2,
            "torque_max_abs": torque_max_abs,
            "torque_limit_max": torque_limit_max,
            "torque_sat_ratio": torque_sat_ratio,
        }
        mpc_diag.update(self._last_solver_diag)

        qp_status = mpc_diag.get('qp_status', np.nan)
        solve_status = int(mpc_diag.get('solve_status', -999))
        minstep_detected = bool(
            (isinstance(qp_status, (int, float)) and np.isfinite(qp_status) and int(qp_status) == 3)
            or solve_status == 3
            or solve_status == 4
        )
        maxiter_detected = bool(solve_status == 2)
        failure_detected = bool(solve_status in (3, 4))
        if maxiter_detected:
            self._acados_maxiter_count_total += 1
            self._acados_maxiter_count_episode += 1
        if minstep_detected:
            self._acados_minstep_count_total += 1
            self._acados_minstep_count_episode += 1
        if failure_detected:
            self._acados_failure_count_total += 1
            self._acados_failure_count_episode += 1

        mpc_diag['acados_minstep_detected'] = float(1.0 if minstep_detected else 0.0)
        mpc_diag['acados_minstep_count_episode'] = float(self._acados_minstep_count_episode)
        mpc_diag['acados_minstep_count_total'] = float(self._acados_minstep_count_total)
        mpc_diag['acados_maxiter_detected'] = float(1.0 if maxiter_detected else 0.0)
        mpc_diag['acados_maxiter_count_episode'] = float(self._acados_maxiter_count_episode)
        mpc_diag['acados_maxiter_count_total'] = float(self._acados_maxiter_count_total)
        mpc_diag['acados_failure_detected'] = float(1.0 if failure_detected else 0.0)
        mpc_diag['acados_failure_count_episode'] = float(self._acados_failure_count_episode)
        mpc_diag['acados_failure_count_total'] = float(self._acados_failure_count_total)

        if self._instrumentation_enabled:
            self._instrumentation_step_counter += 1
            if self._instrumentation_step_counter % self._instrumentation_print_interval == 0:
                print(
                    "[IsaacMPCDiag] status={} dt={:.2f}ms tau_max={:.2f} sat={:.2f} sqp={} qp={} kkt={:.2e}".format(
                        mpc_diag.get('solve_status', -999),
                        mpc_diag.get('compute_duration_ms', 0.0),
                        mpc_diag.get('torque_max_abs', 0.0),
                        mpc_diag.get('torque_sat_ratio', 0.0),
                        mpc_diag.get('sqp_iter', -1),
                        mpc_diag.get('qp_iter', -1),
                        mpc_diag.get('kkt_norm_inf', np.nan),
                    )
                )

        # Pull live values directly from the active swing trajectory generator.
        swing_generator = getattr(getattr(self.wb_interface, "stc", None), "swing_generator", None)
        swing_generator_step_height = float(
            getattr(swing_generator, "step_height", getattr(swing_generator, "stepHeight", np.nan))
        )
        swing_generator_swing_period = float(getattr(swing_generator, "swing_period", np.nan))
        
        debug_info = {
            "mpc_freq": self.best_sample_freq,
            "contact_sequence": contact_sequence,
            "phase": self.wb_interface.pgg.phase_signal,
            "obstacle_slowdown_active": self.wb_interface.obstacle_slowdown_active,
            "obstacle_slowdown_current_speed": self.wb_interface.obstacle_slowdown_current_speed,
            "obstacle_slowdown_prev_speed": self.wb_interface.obstacle_slowdown_prev_speed,
            "step_height_effective": float(step_height_effective),
            "step_height_nominal": float(self._nominal_step_height),
            "step_height_residual": float(self._rl_step_height_residual_m),
            "swing_generator_step_height": float(swing_generator_step_height),
            "swing_generator_swing_period": float(swing_generator_swing_period),
            "mpc_diag": mpc_diag,
        }

        try:
            # Keep RL-facing reference aligned with the nominal command (pre
            # terrain-pitch adjustment), while exposing effective reference for
            # diagnostics.
            ref_lin_effective = np.asarray(self.ref_base_lin_vel).copy()
            ref_lin_nominal = np.asarray(self.ref_base_lin_vel).copy()
            if hasattr(self, 'wb_interface') and self.wb_interface is not None:
                if getattr(self.wb_interface, 'last_ref_base_lin_vel', None) is not None:
                    ref_lin_effective = np.asarray(self.wb_interface.last_ref_base_lin_vel).copy()
                if getattr(self.wb_interface, 'last_ref_base_lin_vel_nominal', None) is not None:
                    ref_lin_nominal = np.asarray(self.wb_interface.last_ref_base_lin_vel_nominal).copy()

            self._last_rl_snapshot = {
                'position': np.asarray(position).copy(),
                'base_ori_euler_xyz': np.asarray(base_ori_euler_xyz).copy(),
                'linear_velocity': np.asarray(linear_velocity).copy(),
                'base_ang_vel_B': np.asarray(base_ang_vel_B).copy(),
                'base_ang_vel_body_xyz': np.asarray(base_ang_vel_B).copy(),
                'joint_positions': np.asarray(joint_positions).copy(),
                'joint_velocities': np.asarray(joint_velocities).copy(),
                'desired_joint_positions': np.concatenate(pd_target_joints_pos.to_list()).flatten().copy(),
                'desired_joint_velocities': np.concatenate(pd_target_joints_vel.to_list()).flatten().copy(),
                'ref_base_lin_vel': ref_lin_nominal,
                'ref_base_lin_vel_nominal': ref_lin_nominal,
                'ref_base_lin_vel_effective': ref_lin_effective,
                'ref_base_ang_vel': np.asarray(self.ref_base_ang_vel).copy(),
                'current_contact': np.asarray(self.wb_interface.current_contact).copy(),
                'measured_contact_forces': np.asarray(contact_forces if contact_forces is not None else np.zeros(4)).copy(),
                'step_height_effective': float(step_height_effective),
                'adaptive_swing_time': float(getattr(self.wb_interface, 'adaptive_swing_time', self._rl_obs_swing_time)),
                'swing_generator_step_height': float(swing_generator_step_height),
                'swing_generator_swing_period': float(swing_generator_swing_period),
                'feet_pos_w': np.asarray(feet_positions).copy(),
                'foothold_error_body': np.zeros(12, dtype=np.float32),
            }

            # Planned-vs-current foothold tracking error in yaw-aligned body frame.
            try:
                yaw = float(base_ori_euler_xyz[2])
                cy, sy = np.cos(yaw), np.sin(yaw)
                # world -> body yaw frame rotation
                r_bw = np.array([[cy, sy], [-sy, cy]], dtype=np.float32)
                foothold_err_body = np.zeros(12, dtype=np.float32)
                for leg_i, leg in enumerate(self.legs_order):
                    planned_w = np.asarray(self.nmpc_footholds[leg], dtype=np.float32).reshape(-1)
                    curr_w = np.asarray(feet_positions[leg_i * 3:(leg_i + 1) * 3], dtype=np.float32).reshape(-1)
                    if planned_w.size < 3 or curr_w.size < 3:
                        continue
                    err_w = planned_w[:3] - curr_w[:3]
                    err_xy_body = r_bw @ err_w[:2]
                    foothold_err_body[leg_i * 3 + 0] = float(err_xy_body[0])
                    foothold_err_body[leg_i * 3 + 1] = float(err_xy_body[1])
                    foothold_err_body[leg_i * 3 + 2] = float(err_w[2])
                self._last_rl_snapshot['foothold_error_body'] = foothold_err_body
            except Exception:
                pass
        except Exception:
            pass
        
        return torques, debug_info

    def _collect_solver_diagnostics(self) -> Dict[str, float]:
        diag = {
            "sqp_iter": -1,
            "qp_iter": -1,
            "qp_status": -1,
            "kkt_norm_inf": np.nan,
            "acados_time_tot_ms": np.nan,
        }

        try:
            controller = getattr(self.srbd_controller_interface, 'controller', None)
            solver = getattr(controller, 'acados_ocp_solver', None)
            if solver is None:
                return diag

            def _safe_get_stat(name):
                try:
                    value = solver.get_stats(name)
                    if isinstance(value, (list, tuple, np.ndarray)):
                        arr = np.asarray(value).reshape(-1)
                        if arr.size == 0:
                            return np.nan
                        return float(arr[-1])
                    return float(value)
                except Exception:
                    return np.nan

            sqp_iter = _safe_get_stat('sqp_iter')
            qp_iter = _safe_get_stat('qp_iter')
            qp_status = _safe_get_stat('qp_stat')
            kkt_norm_inf = _safe_get_stat('res_stat')
            time_tot_s = _safe_get_stat('time_tot')

            if np.isfinite(sqp_iter):
                diag['sqp_iter'] = int(sqp_iter)
            if np.isfinite(qp_iter):
                diag['qp_iter'] = int(qp_iter)
            if np.isfinite(qp_status):
                diag['qp_status'] = int(qp_status)
            if np.isfinite(kkt_norm_inf):
                diag['kkt_norm_inf'] = float(kkt_norm_inf)
            if np.isfinite(time_tot_s):
                diag['acados_time_tot_ms'] = float(time_tot_s * 1e3)
        except Exception:
            return diag

        return diag
    
    def set_reference_velocity(self, linear_vel: np.ndarray, angular_vel: np.ndarray):
        """Update reference velocities (e.g., from joystick)."""
        self.ref_base_lin_vel = linear_vel.copy()
        self.ref_base_ang_vel = angular_vel.copy()

    def set_reference_pitch_offset(self, pitch_offset_rad: float):
        """Update desired body pitch offset (radians), added to terrain pitch reference."""
        self.ref_base_pitch_offset = float(np.clip(pitch_offset_rad, -0.45, 0.45))

    def _visualize_heightmaps(self, base_pos: np.ndarray, base_yaw: float, ref_state: dict) -> None:
        """Visualize heightmaps on-stage (Omniverse USD) or fall back to PNG.

        - Preferred: create/update per-leg USD Mesh prims at `/World/Heightmaps/<leg>`.
        - Fallback: write a PNG to `/tmp/heightmap_vis_<ts>.png`.
        """
        if getattr(self, 'heightmaps', None) is None:
            return

        # Rate-limit visualization to avoid expensive updates
        vis_interval = float(cfg.simulation_params.get('heightmap_vis_interval', 0.5))
        now = time.time()
        if getattr(self, '_last_heightmap_vis_time', 0.0) + vis_interval > now:
            return
        self._last_heightmap_vis_time = now

        # Try USD on-stage visualization first
        try:
            from pxr import UsdGeom, Gf
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                raise RuntimeError("No USD stage available")

            for leg in ['FL', 'FR', 'RL', 'RR']:
                hm = self.heightmaps[leg]
                if hm is None or getattr(hm, 'data', None) is None:
                    continue
                data = hm.data  # rows x cols x 1 x 3
                rows = int(data.shape[0])
                cols = int(data.shape[1])

                z_values = data[:, :, 0, 2].astype(np.float64)
                finite_mask = np.isfinite(z_values)
                if not np.any(finite_mask):
                    continue
                z_median = float(np.median(z_values[finite_mask]))
                vis_clip_enabled = bool(cfg.simulation_params.get('heightmap_vis_z_clip_enabled', True))
                if vis_clip_enabled:
                    clip_above = float(cfg.simulation_params.get('heightmap_vis_z_clip_above', 0.10))
                    clip_below = float(cfg.simulation_params.get('heightmap_vis_z_clip_below', 0.10))
                    z_min = z_median - clip_below
                    z_max = z_median + clip_above
                else:
                    z_min = -np.inf
                    z_max = np.inf

                # Build points list from absolute positions in the data
                points = []
                for r in range(rows):
                    for c in range(cols):
                        p = data[r, c, 0]
                        z = float(p[2]) if np.isfinite(p[2]) else z_median
                        z = float(np.clip(z, z_min, z_max))
                        points.append(Gf.Vec3f(float(p[0]), float(p[1]), z))

                # Build triangles (two per quad)
                faceVertexCounts = []
                faceVertexIndices = []
                def idx(r, c):
                    return r * cols + c

                for r in range(rows - 1):
                    for c in range(cols - 1):
                        v0 = idx(r, c)
                        v1 = idx(r, c + 1)
                        v2 = idx(r + 1, c + 1)
                        v3 = idx(r + 1, c)
                        faceVertexCounts.append(3)
                        faceVertexIndices.extend([v0, v1, v2])
                        faceVertexCounts.append(3)
                        faceVertexIndices.extend([v0, v2, v3])

                prim_path = f"/World/Heightmaps/{leg}"
                prim = stage.GetPrimAtPath(prim_path)
                if not prim:
                    prim = stage.DefinePrim(prim_path, "Mesh")

                mesh = UsdGeom.Mesh(prim)
                mesh.CreatePointsAttr(points)
                mesh.CreateFaceVertexCountsAttr(faceVertexCounts)
                mesh.CreateFaceVertexIndicesAttr(faceVertexIndices)

                # Optionally set extent
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                zs = [p[2] for p in points]
                mesh.CreateExtentAttr([Gf.Vec3f(min(xs), min(ys), min(zs)), Gf.Vec3f(max(xs), max(ys), max(zs))])

            # Don't Save the stage each frame; user can save from UI if desired.
            return
        except Exception:
            # USD unavailable or failed; fall back to PNG visualizer
            pass

        # No PNG fallback: USD visualization only in simplified mode.
        return

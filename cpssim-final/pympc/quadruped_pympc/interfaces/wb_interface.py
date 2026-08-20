#this version was edited by mena kuzman to fix the ik problem that was causing bad walking behavior
import copy
import time

import numpy as np
from quadruped_pympc.helpers.legs_attr import LegsAttr
from scipy.spatial.transform import Rotation as R

from quadruped_pympc import config as cfg
from quadruped_pympc.helpers.foothold_reference_generator import FootholdReferenceGenerator
#from quadruped_pympc.helpers.inverse_kinematics.inverse_kinematics_numeric_adam import InverseKinematicsNumeric
#from quadruped_pympc.helpers.inverse_kinematics.inverse_kinematics_numeric_mujoco import InverseKinematicsNumeric
from quadruped_pympc.helpers.periodic_gait_generator import PeriodicGaitGenerator
from quadruped_pympc.helpers.swing_trajectory_controller import SwingTrajectoryController
from quadruped_pympc.helpers.terrain_estimator import TerrainEstimator
from quadruped_pympc.helpers.velocity_modulator import VelocityModulator
from quadruped_pympc.helpers.early_stance_detector import EarlyStanceDetector

if cfg.simulation_params['visual_foothold_adaptation'] != 'blind':
    from quadruped_pympc.helpers.visual_foothold_adaptation import VisualFootholdAdaptation


class WBInterface:
    """
    WBInterface is responsible for interfacing with the whole body controller of a quadruped robot.
    It initializes the necessary components for motion planning and control, including gait generation,
    swing trajectory control, and terrain estimation.
    """

    def __init__(self,
                 initial_feet_pos: LegsAttr,
                 legs_order: tuple[str, str, str, str] = ('FL', 'FR', 'RL', 'RR'),
                 feet_geom_id : LegsAttr = None):
        """Constructor of the WBInterface class

        Args:
            initial_feet_pos (LegsAttr): initial feet positions, otherwise they will be all zero
            legs_order (tuple[str, str, str, str], optional): order of the leg. Defaults to ('FL', 'FR', 'RL', 'RR').
        """

        mpc_dt = cfg.mpc_params['dt']
        horizon = cfg.mpc_params['horizon']
        self.legs_order = legs_order
        self.feet_geom_id = feet_geom_id

        # Periodic gait generator --------------------------------------------------------------
        gait_name = cfg.simulation_params['gait']
        gait_params = cfg.simulation_params['gait_params'][gait_name]
        gait_type, duty_factor, step_frequency = (
            gait_params['type'],
            gait_params['duty_factor'],
            gait_params['step_freq'],
        )
        # Given the possibility to use nonuniform discretization,
        # we generate a contact sequence two times longer and with a dt half of the one of the mpc
        self.pgg = PeriodicGaitGenerator(
            duty_factor=duty_factor, step_freq=step_frequency, gait_type=gait_type, horizon=horizon
        )
        # in the case of nonuniform discretization, we create a list of dts and horizons for each nonuniform discretization
        if cfg.mpc_params['use_nonuniform_discretization']:
            self.contact_sequence_dts = [cfg.mpc_params['dt_fine_grained'], mpc_dt]
            self.contact_sequence_lenghts = [cfg.mpc_params['horizon_fine_grained'], horizon]
        else:
            self.contact_sequence_dts = [mpc_dt]
            self.contact_sequence_lenghts = [horizon]

        # Create the foothold reference generator ------------------------------------------------
        stance_time = (1 / self.pgg.step_freq) * self.pgg.duty_factor
        self.frg = FootholdReferenceGenerator(
            stance_time=stance_time, hip_height=cfg.hip_height, lift_off_positions=initial_feet_pos
        )

        # Create swing trajectory generator ------------------------------------------------------
        self.step_height = gait_params.get('step_height', cfg.simulation_params['step_height'])
        self.adaptive_swing_time = float(cfg.simulation_params.get('adaptive_swing_time', 0.20))
        self.adaptive_swing_time_min = float(cfg.simulation_params.get('adaptive_swing_time_min', 0.16))
        self.adaptive_swing_time_max = float(cfg.simulation_params.get('adaptive_swing_time_max', 0.22))
        self.adaptive_swing_time_smoothing = float(cfg.simulation_params.get('adaptive_swing_time_smoothing', 0.25))
        self.adaptive_swing_time = float(np.clip(self.adaptive_swing_time, self.adaptive_swing_time_min, self.adaptive_swing_time_max))
        self.frg.swing_time = float(self.adaptive_swing_time)
        swing_period = (1 - self.pgg.duty_factor) * (1 / self.pgg.step_freq)
        position_gain_fb = cfg.simulation_params['swing_position_gain_fb']
        velocity_gain_fb = cfg.simulation_params['swing_velocity_gain_fb']
        swing_generator = cfg.simulation_params['swing_generator']
        self.stc = SwingTrajectoryController(
            step_height=self.step_height,
            swing_period=swing_period,
            position_gain_fb=position_gain_fb,
            velocity_gain_fb=velocity_gain_fb,
            generator=swing_generator,
        )
        self.last_swing_period = swing_period
        self._last_regenerated_step_height = float(self.step_height)
        self.last_des_foot_pos = LegsAttr(*[np.zeros((3,)) for _ in range(4)])
        

        # Terrain estimator -----------------------------------------------------------------------
        self.terrain_computation = TerrainEstimator()

        # Inverse Kinematics ---------------------------------------------------------------------
        #self.ik = InverseKinematicsNumeric()

        if cfg.simulation_params['visual_foothold_adaptation'] != 'blind':
            # Visual foothold adaptation -------------------------------------------------------------
            self.vfa = VisualFootholdAdaptation(
                legs_order=self.legs_order, adaptation_strategy=cfg.simulation_params['visual_foothold_adaptation']
            )

        # Velocity modulator ---------------------------------------------------------------------
        self.vm = VelocityModulator()

        # Early Stance detector -------------------------------------------------------------------
        self.esd = EarlyStanceDetector(feet_geom_id)

        self.current_contact = np.array([1, 1, 1, 1])
        self.previous_contact = np.array([1, 1, 1, 1])

        # Obstacle-adaptive step height state tracking (per-leg) --------------------------------
        self.obstacle_retry_boost = {leg: 0.0 for leg in self.legs_order}  # Current height boost per leg
        self.early_stance_hit_count = {leg: 0 for leg in self.legs_order}  # Hit count per leg during current swing
        self.last_hit_swing_cycle = {leg: -1 for leg in self.legs_order}   # Swing cycle when last hit occurred
        self.obstacle_retry_height = {leg: 0.0 for leg in self.legs_order}  # Incremental height for reflex replan per leg
        self.swing_cycle_counter = 0  # Counter to track swing cycles for obstacle retry logic
        self.last_ref_base_lin_vel = None
        self.last_ref_base_lin_vel_nominal = None
        self.terrain_height_during_boost = {leg: None for leg in self.legs_order}  # Track terrain height when boost is applied
        self.gait_cycle_counter = 0  # Track total gait cycles (one full gait = all 4 legs complete swing-stance)
        self.gait_cycle_at_boost = {leg: 0 for leg in self.legs_order}  # Gait cycle when boost was applied
        self.last_heightmaps = None
        self.last_ref_feet_pos = None

        # Obstacle-triggered speed recovery -----------------------------------------------------
        self.obstacle_slowdown_active = False
        self.obstacle_slowdown_prev_speed = 0.0
        self.obstacle_slowdown_current_speed = None
        self.height_obstacle_prev_detected = False
        self.blind_obstacle_prev_detected = False
        self.last_terrain_height = None  # Track terrain height for ramp rate detection
        self.last_terrain_pitch = 0.0  # Track terrain pitch for downslope detection
        self.debug_log_counter = 0
        self.terrain_stable_cycles = 0
        self.reference_pitch_offset = 0.0

    def reset_for_new_episode(self):
        """Reset episode-dependent internal state while preserving configuration."""
        try:
            self.pgg.reset()
        except Exception:
            pass

        self.current_contact = np.array([1, 1, 1, 1])
        self.previous_contact = np.array([1, 1, 1, 1])

        self.obstacle_slowdown_active = False
        self.obstacle_slowdown_prev_speed = 0.0
        self.obstacle_slowdown_current_speed = None
        self.height_obstacle_prev_detected = False
        self.blind_obstacle_prev_detected = False
        self.last_terrain_height = None
        self.last_terrain_pitch = 0.0
        self.terrain_stable_cycles = 0

        self.gait_cycle_counter = 0
        self.swing_cycle_counter = 0
        self.set_adaptive_swing_time(float(cfg.simulation_params.get('adaptive_swing_time', 0.20)))

        for leg in self.legs_order:
            self.obstacle_retry_boost[leg] = 0.0
            self.early_stance_hit_count[leg] = 0
            self.last_hit_swing_cycle[leg] = -1
            self.obstacle_retry_height[leg] = 0.0
            self.terrain_height_during_boost[leg] = None
            self.gait_cycle_at_boost[leg] = 0

    def update_state_and_reference(
        self,
        com_pos: np.ndarray,
        base_pos: np.ndarray,
        base_lin_vel: np.ndarray,
        base_ori_euler_xyz: np.ndarray,
        base_ang_vel: np.ndarray,
        feet_pos: LegsAttr,
        hip_pos: LegsAttr,
        joints_pos: LegsAttr,
        heightmaps,
        legs_order: tuple[str, str, str, str],
        simulation_dt: float,
        ref_base_lin_vel: np.ndarray,
        ref_base_ang_vel: np.ndarray,
        mujoco_contact: np.ndarray = None,
    ) -> tuple[dict, dict, list, LegsAttr, float, bool]:
        """Update the state and reference for the whole body controller, including the contact sequence, footholds, and terrain estimation.

        Args:
            com_pos (np.ndarray): center of mass position in world frame
            base_pos (np.ndarray): base position in world frame
            base_lin_vel (np.ndarray): base linear velocity in world frame
            base_ori_euler_xyz (np.ndarray): base orientation in euler angles in world frame
            base_ang_vel (np.ndarray): base angular velocity in base frame
            feet_pos (LegsAttr): feet positions in world frame
            hip_pos (LegsAttr): hip positions in world frame
            joints_pos (LegsAttr): joint positions
            heightmaps (dict): heightmaps for each leg
            legs_order (tuple[str, str, str, str]): order of the legs
            simulation_dt (float): simulation time step
            ref_base_lin_vel (np.ndarray): reference base linear velocity in world frame
            ref_base_ang_vel (np.ndarray): reference base angular velocity in world frame

        Returns:
            state_current (dict): dictionary of the state of the robot that is used in the mpc
            ref_state (dict):  dictionary of the reference state of the robot that is used in the mpc
            contact_sequence (np.ndarray): this is an array, containing the contact sequence of the robot in the future
            ref_feet_pos (LegsAttr): where to step in world frame
            ref_feet_constraints (LegsAttr): constraints for the footholds in the world frame
            step_height (float): step height
            optimize_swing (bool), boolean to inform that the robot is in the apex, hence we can optimize step freq.
        """

        state_current = dict(
            position=com_pos + self.frg.com_pos_offset_w,  # manual com offset
            # position=base_pos,
            linear_velocity=base_lin_vel,
            orientation=base_ori_euler_xyz,
            angular_velocity=base_ang_vel,
            foot_FL=feet_pos.FL,
            foot_FR=feet_pos.FR,
            foot_RL=feet_pos.RL,
            foot_RR=feet_pos.RR,
            joint_FL=joints_pos.FL,
            joint_FR=joints_pos.FR,
            joint_RL=joints_pos.RL,
            joint_RR=joints_pos.RR,
        )

        # Modulate the desired velocity if the robot is in strange positions
        if self.vm.activated:
            ref_base_lin_vel, ref_base_ang_vel = self.vm.modulate_velocities(
                ref_base_lin_vel, ref_base_ang_vel, feet_pos, hip_pos
            )


        self.last_ref_base_lin_vel_nominal = ref_base_lin_vel.copy()

        # Estimate the terrain slope, height, and robot elevation -------------------------------------------------------
        terrain_roll, terrain_pitch, terrain_height = self.terrain_computation.compute_terrain_estimation(
            base_position=base_pos,
            yaw=base_ori_euler_xyz[2],
            feet_pos=self.frg.lift_off_positions,
            current_contact=self.current_contact,
        )

        terrain_height_tolerance = cfg.simulation_params.get('obstacle_slowdown_terrain_height_tolerance', 0.01)
        terrain_pitch_tolerance = cfg.simulation_params.get('obstacle_slowdown_terrain_pitch_tolerance', 0.01)

        if hasattr(self, 'last_terrain_pitch') and self.last_terrain_pitch is not None and abs(terrain_pitch - self.last_terrain_pitch) < terrain_pitch_tolerance:
            self.terrain_stable_cycles += 1
        else:
            self.terrain_stable_cycles = 0

        self.last_terrain_pitch = terrain_pitch

        if cfg.simulation_params.get('obstacle_slowdown_enabled', False) and self.obstacle_slowdown_active:
            speed_threshold = cfg.simulation_params.get('obstacle_slowdown_speed_threshold', 1.0)
            low_speed_percentage = cfg.simulation_params.get('obstacle_slowdown_low_speed_percentage', 0.4)
            high_speed_percentage = cfg.simulation_params.get('obstacle_slowdown_high_speed_percentage', 0.6)
            ramp_rate = cfg.simulation_params.get('obstacle_slowdown_ramp_rate', 0.3)
            stable_cycles_required = cfg.simulation_params.get('obstacle_slowdown_stable_cycles', 15)
            ramp_enabled = cfg.simulation_params.get('obstacle_slowdown_ramp_enabled', False)
            
            # Select percentage based on pre-collision speed
            if self.obstacle_slowdown_prev_speed <= speed_threshold:
                slowdown_percentage = low_speed_percentage
            else:
                slowdown_percentage = high_speed_percentage
            
            # Target speed is a percentage of the pre-collision speed
            target_speed = self.obstacle_slowdown_prev_speed * slowdown_percentage

            if self.obstacle_slowdown_current_speed is None:
                self.obstacle_slowdown_current_speed = target_speed
            else:
                # Only apply ramp if enabled and terrain pitch is constant (stable)
                if ramp_enabled and self.terrain_stable_cycles >= stable_cycles_required:
                    self.obstacle_slowdown_current_speed = min(
                        self.obstacle_slowdown_current_speed + ramp_rate * simulation_dt,
                        self.obstacle_slowdown_prev_speed,
                    )

            if self.obstacle_slowdown_current_speed >= self.obstacle_slowdown_prev_speed:
                self.obstacle_slowdown_active = False
                self.obstacle_slowdown_current_speed = None

            ref_speed = float(np.linalg.norm(ref_base_lin_vel[0:2]))
            if ref_speed > 1e-6:
                ref_base_lin_vel[0:2] = (ref_base_lin_vel[0:2] / ref_speed) * min(
                    self.obstacle_slowdown_current_speed or target_speed, ref_speed
                )
            else:
                ref_base_lin_vel[0:2] = 0.0

        # Update the desired contact sequence ---------------------------
        if self.pgg.start_and_stop_activated:
            # stop the robot for energy efficency if there is no movement and its safe
            # only if activated by an an internal flag for now
            self.pgg.update_start_and_stop(
                feet_pos,
                hip_pos,
                self.frg.hip_offset,
                base_pos,
                base_ori_euler_xyz,
                base_lin_vel,
                base_ang_vel,
                ref_base_lin_vel,
                ref_base_ang_vel,
                self.current_contact,
            )

        if cfg.simulation_params.get('adaptive_gait_enabled', False):
            yaw = base_ori_euler_xyz[2]
            rotation_world_to_horizontal = np.array([np.cos(yaw), np.sin(yaw), -np.sin(yaw), np.cos(yaw)]).reshape((2, 2))
            base_lin_vel_h = rotation_world_to_horizontal @ base_lin_vel[0:2]
            ref_base_lin_vel_h = rotation_world_to_horizontal @ ref_base_lin_vel[0:2]

            adapted_speed = self.frg.adaptive_speed(base_lin_vel_h, ref_base_lin_vel_h)
            adaptive_activation_speed = cfg.simulation_params.get('adaptive_activation_speed', 0.3)
            
            # Only apply adaptive gait when velocity exceeds activation threshold
            if adapted_speed >= adaptive_activation_speed:
                period = self.frg.adaptive_period(adapted_speed)
                self.frg.swing_time = float(np.clip(self.adaptive_swing_time, self.adaptive_swing_time_min, self.adaptive_swing_time_max))
                duty_factor = self.frg.adaptive_duty_factor(period)

                swing_period = float(self.adaptive_swing_time)
                step_height = self.step_height

                self.pgg.step_freq = 1.0 / period
                self.pgg.duty_factor = duty_factor
                self.frg.stance_time = duty_factor * period

                if abs(swing_period - self.last_swing_period) > 1e-4:
                    self.stc.regenerate_swing_trajectory_generator(step_height, swing_period)
                    self.last_swing_period = swing_period
                    self._last_regenerated_step_height = float(step_height)

        # Ensure the swing generator reflects the current step_height even when
        # the adaptive-gait block did not fire (e.g. low speed, or adaptive gait
        # disabled).  The RL residual can change step_height every cycle, but
        # the SwingTrajectoryGenerator only sees it at regeneration time.
        if abs(float(self.step_height) - self._last_regenerated_step_height) > 1e-3:
            current_swing_period = self.last_swing_period
            self.stc.regenerate_swing_trajectory_generator(float(self.step_height), current_swing_period)
            self._last_regenerated_step_height = float(self.step_height)

        self.pgg.run(simulation_dt, self.pgg.step_freq)
        contact_sequence = self.pgg.compute_contact_sequence(
            contact_sequence_dts=self.contact_sequence_dts, contact_sequence_lenghts=self.contact_sequence_lenghts
        )

        self.previous_contact = copy.deepcopy(self.current_contact)
        # Handle variable number of legs in contact_sequence
        num_legs = contact_sequence.shape[0]  # Get actual number of legs from contact_sequence
        if num_legs == 4:
            self.current_contact = np.array(
                [contact_sequence[0][0], contact_sequence[1][0], contact_sequence[2][0], contact_sequence[3][0]]
            )
        else:
            # Fallback: create array from however many legs we have
            self.current_contact = np.array([contact_sequence[i][0] for i in range(num_legs)])

        # Compute the reference for the footholds ---------------------------------------------------
        self.frg.update_lift_off_positions(
            self.previous_contact,
            self.current_contact,
            feet_pos,
            legs_order,
            self.pgg.gait_type,
            base_pos,
            base_ori_euler_xyz,
        )
        self.frg.update_touch_down_positions(
            self.previous_contact,
            self.current_contact,
            feet_pos,
            legs_order,
            self.pgg.gait_type,
            base_pos,
            base_ori_euler_xyz,
        )
        ref_feet_pos = self.frg.compute_footholds_reference(
            base_position=base_pos,
            base_ori_euler_xyz=base_ori_euler_xyz,
            base_xy_lin_vel=base_lin_vel[0:2],
            ref_base_xy_lin_vel=ref_base_lin_vel[0:2],
            hips_position=hip_pos,
            com_height_nominal=cfg.simulation_params['ref_z'],
        )

        # Adjust the footholds given the terrain -----------------------------------------------------
        if cfg.simulation_params['visual_foothold_adaptation'] != 'blind':
            command_metric = float(np.linalg.norm(ref_base_lin_vel[0:2]) + abs(ref_base_ang_vel[2]))
            disable_at_zero_cmd = cfg.simulation_params.get('heightmap_disable_at_zero_cmd', True)
            zero_cmd_threshold = float(cfg.simulation_params.get('heightmap_zero_cmd_threshold', 0.02))

            skip_heightmap_adaptation = (
                disable_at_zero_cmd
                and command_metric < zero_cmd_threshold
            )

            if skip_heightmap_adaptation:
                if self.vfa.initialized:
                    self.vfa.reset()
                ref_feet_constraints = LegsAttr(FL=None, FR=None, RL=None, RR=None)
            else:
                time_adaptation = time.time()
                # Refresh adaptation every apex so references track changing velocity/terrain.
                if self.stc.check_apex_condition(self.current_contact, interval=0.01):
                    for leg_id, leg_name in enumerate(legs_order):
                        heightmaps[leg_name].update_height_map(ref_feet_pos[leg_name], yaw=base_ori_euler_xyz[2])
                    self.vfa.compute_adaptation(
                        legs_order, ref_feet_pos, hip_pos, heightmaps, base_lin_vel, base_ori_euler_xyz, base_ang_vel
                    )
                    # print("Adaptation time: ", time.time() - time_adaptation)

                if self.stc.check_full_stance_condition(self.current_contact):
                    self.vfa.reset()

                ref_feet_pos, ref_feet_constraints = self.vfa.get_footholds_adapted(ref_feet_pos)
        else:
            ref_feet_constraints = LegsAttr(FL=None, FR=None, RL=None, RR=None)

        ref_pos = np.array([0, 0, cfg.hip_height])
        ref_pos[2] = cfg.simulation_params["ref_z"] + terrain_height
        
        # Rotate the reference base linear velocity to the terrain frame
        ref_base_lin_vel = R.from_euler("xyz", [terrain_roll, terrain_pitch, 0]).as_matrix() @ ref_base_lin_vel
        if(terrain_pitch > 0.0):
            ref_base_lin_vel[2] = -ref_base_lin_vel[2]

        # Optional steep-slope speed limiting (disabled by default).
        # This was previously always-on and could cap flat-ground speed when terrain
        # pitch estimates became noisy due to foothold adaptation updates.
        if cfg.simulation_params.get('terrain_pitch_speed_limit_enabled', False):
            pitch_limit_threshold = float(cfg.simulation_params.get('terrain_pitch_speed_limit_threshold', 0.2))
            speed_limit_factor = float(cfg.simulation_params.get('terrain_pitch_speed_limit_factor', 0.5))
            vertical_scale = float(cfg.simulation_params.get('terrain_pitch_vertical_scale', 2.0))
            if np.abs(terrain_pitch) > pitch_limit_threshold:
                ref_base_lin_vel[0] = ref_base_lin_vel[0] * speed_limit_factor
                ref_base_lin_vel[2] = ref_base_lin_vel[2] * vertical_scale

        # Update state reference ------------------------------------------------------------------------

        # Since the MPC close in CoM position, but usually we have desired height for the base,
        # we modify the reference to bring the base at the desired height and not the CoM
        ref_pos[2] -= base_pos[2] - (com_pos[2] + self.frg.com_pos_offset_w[2])

        if cfg.mpc_params['type'] != 'kinodynamic':
            ref_state = {}
            ref_state |= dict(
                ref_foot_FL=ref_feet_pos.FL.reshape((1, 3)),
                ref_foot_FR=ref_feet_pos.FR.reshape((1, 3)),
                ref_foot_RL=ref_feet_pos.RL.reshape((1, 3)),
                ref_foot_RR=ref_feet_pos.RR.reshape((1, 3)),
                ref_foot_constraints_FL=ref_feet_constraints.FL,
                ref_foot_constraints_FR=ref_feet_constraints.FR,
                ref_foot_constraints_RL=ref_feet_constraints.RL,
                ref_foot_constraints_RR=ref_feet_constraints.RR,
                # Also update the reference base linear velocity and
                ref_linear_velocity=ref_base_lin_vel,
                ref_angular_velocity=ref_base_ang_vel,
                ref_orientation=np.array([terrain_roll, terrain_pitch + float(self.reference_pitch_offset), 0.0]),
                ref_position=ref_pos,
            )

        # -------------------------------------------------------------------------------------------------

        if cfg.mpc_params['optimize_step_freq']:
            # we can always optimize the step freq, or just at the apex of the swing
            # to avoid possible jittering in the solution
            optimize_swing = self.stc.check_touch_down_condition(self.current_contact, self.previous_contact, contact_sequence, lookahead=3)
        else:
            optimize_swing = 0

        self.last_ref_base_lin_vel = ref_base_lin_vel.copy()
        self.last_terrain_height = terrain_height
        self.last_terrain_pitch = terrain_pitch
        self.last_heightmaps = heightmaps
        self.last_ref_feet_pos = copy.deepcopy(ref_feet_pos)

        return state_current, ref_state, contact_sequence, self.step_height, optimize_swing

    def set_adaptive_swing_time(self, swing_time: float) -> None:
        swing_time = float(np.clip(swing_time, self.adaptive_swing_time_min, self.adaptive_swing_time_max))
        self.adaptive_swing_time = float(swing_time)
        self.frg.swing_time = float(self.adaptive_swing_time)

    def compute_stance_and_swing_torque(
        self,
        simulation_dt: float,
        qpos: np.ndarray,
        qvel: np.ndarray,
        feet_jac: LegsAttr,
        feet_jac_dot: LegsAttr,
        feet_pos: LegsAttr,
        feet_vel: LegsAttr,
        legs_qfrc_passive: LegsAttr,
        legs_qfrc_bias: LegsAttr,
        legs_mass_matrix: LegsAttr,
        nmpc_GRFs: LegsAttr,
        nmpc_footholds: LegsAttr,
        legs_qpos_idx: LegsAttr,
        legs_qvel_idx: LegsAttr,
        tau: LegsAttr,
        optimize_swing: int,
        best_sample_freq: float,
        nmpc_joints_pos,
        nmpc_joints_vel,
        nmpc_joints_acc,
        nmpc_predicted_state,
        mujoco_contact: np.ndarray = None,
    ) -> LegsAttr:
        """Compute the stance and swing torque.

        Args:
            simulation_dt (float): simulation time step
            qvel (np.ndarray): joint velocities
            feet_jac (LegsAttr): feet jacobian
            feet_jac_dot (LegsAttr): derivative of the jacobian
            feet_pos (LegsAttr): feet positions in world frame
            feet_vel (LegsAttr): feet velocities in world frame
            legs_qfrc_passive (LegsAttr): passive forces and torques
            legs_qfrc_bias (LegsAttr): joint forces and torques
            legs_mass_matrix (LegsAttr): mass matrix of the legs
            nmpc_GRFs (LegsAttr): ground reaction forces from the MPC in world frame
            nmpc_footholds (LegsAttr): footholds from the MPC in world frame
            legs_qvel_idx (LegsAttr): joint velocities index
            tau (LegsAttr): joint torques
            optimize_swing (int): flag to signal that we need to update the swing trajectory time
            best_sample_freq (float): best sample frequency obtained from the
                                      sampling optimization or the batched ocp

        Returns:
            LegsAttr: joint torques
        """

        # If we have optimized the gait, we set all the timing parameters
        if optimize_swing == 1:
            self.pgg.step_freq = np.array([best_sample_freq])[0]
            self.frg.stance_time = (1 / self.pgg.step_freq) * self.pgg.duty_factor
            swing_period = (1 - self.pgg.duty_factor) * (1 / self.pgg.step_freq)
            self.stc.regenerate_swing_trajectory_generator(step_height=self.step_height, swing_period=swing_period)
            self._last_regenerated_step_height = float(self.step_height)

        
        # Update the Early Stance Detector for Reflexes
        self.esd.update_detection(
            feet_pos,
            self.last_des_foot_pos,
            lift_off=self.frg.lift_off_positions,
            touch_down=nmpc_footholds,
            swing_time=self.stc.swing_time,
            swing_period=self.stc.swing_period,
            current_contact=self.current_contact,
            previous_contact=self.previous_contact,
            mujoco_contact=mujoco_contact,
            stc=self.stc,
        )



        if cfg.simulation_params.get('debug_contact_logging', False):
            self.debug_log_counter += 1
            stride = cfg.simulation_params.get('debug_contact_logging_stride', 50)
            if stride > 0 and (self.debug_log_counter % stride) == 0:
                early = {leg: bool(self.esd.early_stance[leg]) for leg in self.legs_order}
                contact = {leg: int(self.current_contact[i]) for i, leg in enumerate(self.legs_order)}
                print(f"[contact] cycle={self.debug_log_counter} contact={contact} early_stance={early}")

        mode = cfg.simulation_params.get('visual_foothold_adaptation', 'blind')
        slowdown_enabled = cfg.simulation_params.get('obstacle_slowdown_enabled', False)

        if slowdown_enabled:
            obstacle_detected = False
            trigger_slowdown = False
            
            # Strategy 1: Heightmap-based predictive detection (when heightmaps are enabled)
            if mode == 'height':
                obstacle_detected = self._heightmap_obstacle_detected()
                # One-shot trigger on rising edge only.
                trigger_slowdown = obstacle_detected and (not self.height_obstacle_prev_detected)
                self.height_obstacle_prev_detected = obstacle_detected
            
            # Strategy 2: Early stance reactive detection (when in blind mode)
            if mode == 'blind':
                obstacle_detected = any(self.esd.early_stance[leg] for leg in self.legs_order)
                # One-shot trigger on rising edge only to avoid retriggering every tick.
                trigger_slowdown = obstacle_detected and (not self.blind_obstacle_prev_detected)
                self.blind_obstacle_prev_detected = obstacle_detected
                # Keep height-mode edge detector reset while not in height mode.
                self.height_obstacle_prev_detected = False

            if mode != 'blind':
                self.blind_obstacle_prev_detected = False
            
            # Activate slowdown if obstacle detected by either method
            if trigger_slowdown and not self.obstacle_slowdown_active:
                if self.last_ref_base_lin_vel_nominal is not None:
                    ref_vel = self.last_ref_base_lin_vel_nominal
                elif self.last_ref_base_lin_vel is not None:
                    ref_vel = self.last_ref_base_lin_vel
                else:
                    ref_vel = None
                if ref_vel is not None:
                    self.obstacle_slowdown_prev_speed = float(np.linalg.norm(ref_vel[0:2]))
                else:
                    self.obstacle_slowdown_prev_speed = 0.0

                # Only activate slowdown if current speed exceeds minimum threshold
                min_activation_speed = cfg.simulation_params.get('obstacle_slowdown_min_activation_speed', 0.5)
                if self.obstacle_slowdown_prev_speed > min_activation_speed:
                    self.obstacle_slowdown_active = True
                    self.obstacle_slowdown_current_speed = None
        else:
            if self.obstacle_slowdown_active:
                self.obstacle_slowdown_active = False
                self.obstacle_slowdown_current_speed = None

        # Increment swing cycle counter when any leg transitions from stance to swing
        for leg_id in range(len(self.legs_order)):
            if self.previous_contact[leg_id] == 1 and self.current_contact[leg_id] == 0:
                self.swing_cycle_counter += 1
        
        # Increment gait cycle counter when all legs have completed a swing phase
        # (approximate: increment when front-left leg transitions to swing as a full gait marker)
        if self.previous_contact[0] == 1 and self.current_contact[0] == 0:
            self.gait_cycle_counter += 1

        # Update obstacle retry boost based on early stance hits (for blind robot obstacle climbing)
        self.update_obstacle_retry_boost(self.swing_cycle_counter)


        # Compute Stance Torque ---------------------------------------------------------------------------
        tau.FL = -np.matmul(feet_jac.FL[:, legs_qvel_idx.FL].T, nmpc_GRFs.FL)
        tau.FR = -np.matmul(feet_jac.FR[:, legs_qvel_idx.FR].T, nmpc_GRFs.FR)
        tau.RL = -np.matmul(feet_jac.RL[:, legs_qvel_idx.RL].T, nmpc_GRFs.RL)
        tau.RR = -np.matmul(feet_jac.RR[:, legs_qvel_idx.RR].T, nmpc_GRFs.RR)

        self.stc.update_swing_time(self.current_contact, self.legs_order, simulation_dt)

        # Compute Swing Torque ------------------------------------------------------------------------------
        des_foot_pos = LegsAttr(*[np.zeros((3,)) for _ in range(4)])
        des_foot_vel = LegsAttr(*[np.zeros((3,)) for _ in range(4)])

        # The swing controller is in the end-effector space
        for leg_id, leg_name in enumerate(self.legs_order):
            if (
                self.current_contact[leg_id] == 0
            ):  # If in swing phase, compute the swing trajectory tracking control.
                # Set obstacle retry height for this leg's swing generator
                self.stc.swing_generator.obstacle_retry_height = self.obstacle_retry_height[leg_name]
                
                tau[leg_name], des_foot_pos[leg_name], des_foot_vel[leg_name] = (
                    self.stc.compute_swing_control_cartesian_space(
                        leg_id=leg_id,
                        q_dot=qvel[legs_qvel_idx[leg_name]],
                        J=feet_jac[leg_name][:, legs_qvel_idx[leg_name]],
                        J_dot=feet_jac_dot[leg_name][:, legs_qvel_idx[leg_name]],
                        lift_off=self.frg.lift_off_positions[leg_name],
                        touch_down=nmpc_footholds[leg_name],
                        foot_pos=feet_pos[leg_name],
                        foot_vel=feet_vel[leg_name],
                        passive_force=legs_qfrc_passive[leg_name],
                        h=legs_qfrc_bias[leg_name],
                        mass_matrix=legs_mass_matrix[leg_name],
                        early_stance_hitmoments=self.esd.hitmoments[leg_name],
                        early_stance_hitpoints=self.esd.hitpoints[leg_name],
                    )
                )

            else:
                des_foot_pos[leg_name] = nmpc_footholds[leg_name]
                # des_foot_pos[leg_name] = self.frg.touch_down_positions[leg_name]
                des_foot_vel[leg_name] = des_foot_vel[leg_name] * 0.0


        self.last_des_foot_pos = des_foot_pos

        # Compensate for friction -------------------------------------------------------------
        #self.stc.use_friction_compensation = True
        if(self.stc.use_friction_compensation): #TODO fix this flag, is not only related to swing
            tau.FL -= legs_qfrc_passive.FL
            tau.FR -= legs_qfrc_passive.FR
            tau.RL -= legs_qfrc_passive.RL
            tau.RR -= legs_qfrc_passive.RR


        # Compute PD targets for the joints ----------------------------------------------------------------
        des_joints_pos = LegsAttr(*[np.zeros((3, 1)) for _ in range(4)])
        des_joints_vel = LegsAttr(*[np.zeros((3, 1)) for _ in range(4)])
        # if cfg.mpc_params['type'] != 'kinodynamic':
        #     qpos_predicted = copy.deepcopy(qpos)
        #     # TODO use predicted rotation too
        #     # qpos_predicted[0:3] = nmpc_predicted_state[0:3]
        #     temp = self.ik.compute_solution(
        #         qpos_predicted, des_foot_pos.FL, des_foot_pos.FR, des_foot_pos.RL, des_foot_pos.RR
        #     )

        #     des_joints_pos.FL = np.array(temp[0:3]).reshape((3,))
        #     des_joints_pos.FR = np.array(temp[3:6]).reshape((3,))
        #     des_joints_pos.RL = np.array(temp[6:9]).reshape((3,))
        #     des_joints_pos.RR = np.array(temp[9:12]).reshape((3,))

        #     # TODO This should be done over the the desired joint positions jacobian
        #     des_joints_vel.FL = np.linalg.pinv(feet_jac.FL[:, legs_qvel_idx.FL]) @ des_foot_vel.FL
        #     des_joints_vel.FR = np.linalg.pinv(feet_jac.FR[:, legs_qvel_idx.FR]) @ des_foot_vel.FR
        #     des_joints_vel.RL = np.linalg.pinv(feet_jac.RL[:, legs_qvel_idx.RL]) @ des_foot_vel.RL
        #     des_joints_vel.RR = np.linalg.pinv(feet_jac.RR[:, legs_qvel_idx.RR]) @ des_foot_vel.RR
        # mena kuzman change to fix ik problem
        if cfg.mpc_params['type'] != 'kinodynamic':
            qpos_predicted = copy.deepcopy(qpos)
            # Get current joint positions (from qpos)
            q_joint_FL = qpos[legs_qpos_idx.FL]
            q_joint_FR = qpos[legs_qpos_idx.FR]
            q_joint_RL = qpos[legs_qpos_idx.RL]
            q_joint_RR = qpos[legs_qpos_idx.RR]

            # Get current foot positions (from passed-in Isaac Sim data)
            actual_foot_pos_FL = feet_pos.FL
            actual_foot_pos_FR = feet_pos.FR
            actual_foot_pos_RL = feet_pos.RL
            actual_foot_pos_RR = feet_pos.RR
            
            # Get target foot positions (from swing controller)
            target_foot_pos_FL = des_foot_pos.FL
            target_foot_pos_FR = des_foot_pos.FR
            target_foot_pos_RL = des_foot_pos.RL
            target_foot_pos_RR = des_foot_pos.RR

            # Get the 3x3 joint-space Jacobians (from passed-in Isaac Sim data)
            J_FL = feet_jac.FL[:, legs_qvel_idx.FL]
            J_FR = feet_jac.FR[:, legs_qvel_idx.FR]
            J_RL = feet_jac.RL[:, legs_qvel_idx.RL]
            J_RR = feet_jac.RR[:, legs_qvel_idx.RR]

            # Assemble 12x12 block-diagonal Jacobian
            J_12x12 = np.zeros((12, 12))
            J_12x12[0:3, 0:3]   = J_FL
            J_12x12[3:6, 3:6]   = J_FR
            J_12x12[6:9, 6:9]   = J_RL
            J_12x12[9:12, 9:12] = J_RR

            # Assemble 12x1 error vector
            err_12x1 = np.zeros(12)
            err_12x1[0:3] = target_foot_pos_FL - actual_foot_pos_FL
            err_12x1[3:6] = target_foot_pos_FR - actual_foot_pos_FR
            err_12x1[6:9] = target_foot_pos_RL - actual_foot_pos_RL
            err_12x1[9:12] = target_foot_pos_RR - actual_foot_pos_RR
            
            # Replicating the logic from the IK solver:
            # (damped_pinv @ (100 * err)) * DT
            damp_matrix = (1e-3) * np.eye(12)
            damped_pinv = np.linalg.inv(J_12x12.T @ J_12x12 + damp_matrix) @ J_12x12.T
            dq = (damped_pinv @ (100 * err_12x1)) * (1e-2) # Using DT from old file

            # Set desired joint positions
            des_joints_pos.FL = q_joint_FL + dq[0:3]
            des_joints_pos.FR = q_joint_FR + dq[3:6]
            des_joints_pos.RL = q_joint_RL + dq[6:9]
            des_joints_pos.RR = q_joint_RR + dq[9:12]

            # --- END NEW IK IMPLEMENTATION ---

            # TODO This should be done over the the desired joint positions jacobian
            des_joints_vel.FL = np.linalg.pinv(J_FL) @ des_foot_vel.FL
            des_joints_vel.FR = np.linalg.pinv(J_FR) @ des_foot_vel.FR
            des_joints_vel.RL = np.linalg.pinv(J_RL) @ des_foot_vel.RL
            des_joints_vel.RR = np.linalg.pinv(J_RR) @ des_foot_vel.RR
        else:
            # In the case of the kinodynamic model, we just use the NMPC predicted joints
            des_joints_pos = nmpc_joints_pos
            des_joints_vel = nmpc_joints_vel

        # Saturate of desired joint positions and velocities
        # (only if they were computed; sampling MPC returns None for these)
        if des_joints_pos is not None and des_joints_vel is not None:
            max_joints_pos_difference = 3.0
            max_joints_vel_difference = 10.0

            # Calculate the difference
            actual_joints_pos = LegsAttr(**{leg_name: qpos[legs_qpos_idx[leg_name]] for leg_name in self.legs_order})
            actual_joints_vel = LegsAttr(**{leg_name: qvel[legs_qvel_idx[leg_name]] for leg_name in self.legs_order})

            # Saturate the difference for each leg
            for leg in ["FL", "FR", "RL", "RR"]:
                joints_pos_difference = des_joints_pos[leg] - actual_joints_pos[leg]
                saturated_joints_pos_difference = np.clip(
                    joints_pos_difference, -max_joints_pos_difference, max_joints_pos_difference
                )
                des_joints_pos[leg] = actual_joints_pos[leg] + saturated_joints_pos_difference

                joints_vel_difference = des_joints_vel[leg] - actual_joints_vel[leg]
                saturated_joints_vel_difference = np.clip(
                    joints_vel_difference, -max_joints_vel_difference, max_joints_vel_difference
                )
                des_joints_vel[leg] = actual_joints_vel[leg] + saturated_joints_vel_difference

        return tau, des_joints_pos, des_joints_vel

    def _heightmap_obstacle_detected(self) -> bool:
        if self.last_heightmaps is None or self.last_ref_feet_pos is None:
            return False
        if self.last_terrain_height is None:
            return False

        threshold = cfg.simulation_params.get('obstacle_slowdown_heightmap_threshold', 0.03)
        for leg_id, leg_name in enumerate(self.legs_order):
            if self.current_contact[leg_id] == 1:
                continue
            heightmap = self.last_heightmaps[leg_name]
            if heightmap is None:
                continue
            height_adjustment = heightmap.get_height(self.last_ref_feet_pos[leg_name])
            if height_adjustment is None and getattr(heightmap, 'data', None) is not None:
                height_adjustment = float(np.max(heightmap.data[:, :, 0, 2]))

            if height_adjustment is None:
                continue
            if height_adjustment - self.last_terrain_height > threshold:
                return True
        return False


    def update_obstacle_retry_boost(self, swing_cycle: int):
        """Update per-leg obstacle retry boost state based on early stance detection.
        
        Applies boost increment on obstacle detection.
        Maintains increment while terrain height is changing.
        
        Args:
            swing_cycle: Current swing cycle counter (incremented each swing phase start)
        """
        from quadruped_pympc import config as cfg

        mode = cfg.simulation_params.get('visual_foothold_adaptation', 'blind')
        retry_enabled = cfg.simulation_params.get('obstacle_retry_enabled', False)
        
        if not retry_enabled:
            for leg_name in self.legs_order:
                self.obstacle_retry_boost[leg_name] = 0.0
                self.obstacle_retry_height[leg_name] = 0.0
            return
        
        max_attempts = cfg.simulation_params.get('obstacle_retry_max_attempts', 3)
        terrain_height_tolerance = cfg.simulation_params.get('obstacle_slowdown_terrain_height_tolerance', 0.01)
        
        # Global boost logic: apply boost to all legs if any leg triggers
        boost_triggered = False
        boost_attempt_count = 0
        for leg_id, leg_name in enumerate(self.legs_order):
            is_swinging = self.current_contact[leg_id] == 0
            if is_swinging and self.esd.early_stance[leg_name]:
                if self.last_hit_swing_cycle[leg_name] != swing_cycle:
                    boost_triggered = True
                    boost_attempt_count = max(boost_attempt_count, self.early_stance_hit_count[leg_name] + 1)
                    self.early_stance_hit_count[leg_name] += 1
                    self.last_hit_swing_cycle[leg_name] = swing_cycle

        if boost_triggered and boost_attempt_count <= max_attempts:
            new_boost = self.stc.compute_obstacle_height_boost(attempt_count=boost_attempt_count)
            # Use current step height (already includes RL residual contribution)
            # so retry boost remains additive instead of overriding RL behavior.
            base_height = float(self.step_height)
            for leg_name in self.legs_order:
                self.obstacle_retry_boost[leg_name] = new_boost
                self.terrain_height_during_boost[leg_name] = self.last_terrain_height
                self.obstacle_retry_height[leg_name] = base_height + new_boost
            # Record global boost cycle
            self.global_obstacle_retry_cycle = self.gait_cycle_counter

        # Reset boost for all legs after hold_cycles and stable terrain
        hold_cycles = cfg.simulation_params.get('obstacle_retry_hold_cycles', 5)
        terrain_is_stable = True
        for leg_name in self.legs_order:
            if self.terrain_height_during_boost[leg_name] is not None and self.last_terrain_height is not None:
                terrain_is_stable = terrain_is_stable and (abs(self.last_terrain_height - self.terrain_height_during_boost[leg_name]) < terrain_height_tolerance)

        if hasattr(self, 'global_obstacle_retry_cycle') and self.global_obstacle_retry_cycle is not None:
            gait_cycles_passed = self.gait_cycle_counter - self.global_obstacle_retry_cycle
            if terrain_is_stable and gait_cycles_passed >= hold_cycles:
                for leg_name in self.legs_order:
                    self.obstacle_retry_boost[leg_name] = 0.0
                    self.early_stance_hit_count[leg_name] = 0
                    self.last_hit_swing_cycle[leg_name] = -1
                    self.obstacle_retry_height[leg_name] = 0.0
                    self.terrain_height_during_boost[leg_name] = None
                self.global_obstacle_retry_cycle = None

    def reset(self, initial_feet_pos: LegsAttr):
        """Reset the whole body interface

        Args:
            initial_feet_pos (LegsAttr): initial feet positions
        """

        self.pgg.reset()
        # self.frg.reset()
        # self.stc.reset()
        # self.terrain_computation.reset()
        self.frg.lift_off_positions = initial_feet_pos
        if cfg.simulation_params['visual_foothold_adaptation'] != 'blind':
            self.vfa.reset()
        self.current_contact = np.array([1, 1, 1, 1])
        self.height_obstacle_prev_detected = False
        return

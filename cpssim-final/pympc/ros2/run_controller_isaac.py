import sys
import os
import time
import threading
import multiprocessing
import copy

import rclpy
from rclpy.node import Node
from dls2_msgs.msg import BaseStateMsg, BlindStateMsg, ControlSignalMsg, TrajectoryGeneratorMsg
from sensor_msgs.msg import Joy
from std_msgs.msg import Int32MultiArray  # For jac_map subscriber

import numpy as np
import array
from scipy.spatial.transform import Rotation

# Local application/library specific imports
from quadruped_pympc import config as cfg
print(f"--- [DEBUG] PYTHON IS LOADING CONFIG FROM: {cfg.__file__} ---")
from quadruped_pympc.interfaces.srbd_controller_interface import SRBDControllerInterface
from quadruped_pympc.interfaces.wb_interface import WBInterface
from console import Console  # Keep Console import

np.set_printoptions(precision=3, suppress=True)


# --- Process Priority Settings ---
# Silence module-level prints in this file (avoid noisy stdout during runtime).
# This shadows the builtin `print` only in this module.
def _no_print(*args, **kwargs):
    return None
#print = _no_print

# Set the priority of the process
pid = os.getpid()
try:
    os.nice(-20)
except PermissionError:
    # keep silent on permission issues
    pass
except OSError:
    pass

# --- Configuration Constants ---
USE_DLS_CONVENTION = False # Set True/False based on your robot/sim convention needs, already tested and is false

USE_THREADED_MPC = False
USE_PROCESS_MPC = False # Set only one of THREADED or PROCESS to True
MPC_FREQ = 100.0 # Hz


USE_SCHEDULER = True # This enable a call to the run function every tot seconds, instead of as fast as possible
SCHEDULER_FREQ = 250.0 # Hz (only if USE_SCHEDULER is True)

USE_FIXED_LOOP_TIME = False
USE_SATURATED_LOOP_TIME = True 
MAX_LOOP_TIME_S = 1.0 / 250.0 # Corresponds to 250 Hz cap

USE_SMOOTH_VELOCITY = False
USE_SMOOTH_HEIGHT = True

# Expected controller joint names (fixed order for this controller)
EXPECTED_JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]


# --- Utility Classes ---
class LegsAttr:
    """Dataclass to store attributes associated with the legs of a quadruped robot."""
    def __init__(self, FL, FR, RL, RR):
        self.FL = FL
        self.FR = FR
        self.RL = RL
        self.RR = RR
        self.order = ['FL', 'FR', 'RL', 'RR']

    def to_list(self, order=None):
        order = order if order is not None else self.order
        return [getattr(self, leg) for leg in order]

    def __getitem__(self, key):
        if key not in self.order:
            raise KeyError(f'Key {key} is not a valid leg label. Expected any of {self.order}')
        return getattr(self, key)

    def __setitem__(self, key, value):
        if key not in self.order:
            raise KeyError(f'Key {key} is not a valid leg label. Expected any of {self.order}')
        setattr(self, key, value)

    def __iter__(self):
        return iter(self.to_list())

    def __add__(self, other):
        if isinstance(other, LegsAttr):
            return LegsAttr(
                FL=self.FL + other.FL, FR=self.FR + other.FR,
                RL=self.RL + other.RL, RR=self.RR + other.RR
            )
        elif isinstance(other, (int, float, np.ndarray)):
            return LegsAttr(
                FL=self.FL + other, FR=self.FR + other,
                RL=self.RL + other, RR=self.RR + other
            )
        else:
            return NotImplemented

    def __str__(self):
        parts = []
        for leg in self.order:
            val = getattr(self, leg)
            if isinstance(val, np.ndarray):
                formatted_val = np.array2string(val, precision=3, suppress_small=True, separator=', ')
            else:
                formatted_val = str(val)
            parts.append(f"{leg}={formatted_val}")
        return f'{"; ".join(parts)}'

    def __repr__(self):
        return self.__str__()


# --- ROS2 Node ---
class Quadruped_PyMPC_Node(Node):
    def __init__(self):
        super().__init__('Quadruped_PyMPC_Node')
        # Debug logging flag - when False, all logging calls are no-ops via DummyLogger
        self.ENABLE_DEBUG_LOGGING = False
        try:
            # Store real logger and prepare dummy logger
            self._real_logger = super().get_logger()
        except Exception:
            # On any failure, fallback to default behavior (node's logger)
            pass

        # --- ROS2 Communications ---
        self.subscription_base_state = self.create_subscription(
            BaseStateMsg, "/dls2/base_state", self.get_base_state_callback, 1
        )
        self.subscription_blind_state = self.create_subscription(
            BlindStateMsg, "/dls2/blind_state", self.get_blind_state_callback, 1
        )
        self.subscription_joy = self.create_subscription(
            Joy, "joy", self.get_joy_callback, 1
        )
        self.publisher_control_signal = self.create_publisher(
            ControlSignalMsg, "dls2/quadruped_pympc_torques", 1
        )
        # Keep trajectory publisher
        self.publisher_trajectory_generator = self.create_publisher(
            TrajectoryGeneratorMsg, "dls2/trajectory_generator", 1
        )
        # NOTE: publisher_jac_map REMOVED

        # Subscribe to simulator jacobian map suggestions (one-shot robustness)
        try:
            self.subscription_jac_map = self.create_subscription(
                Int32MultiArray, "/dls2/jacobian_column_map", self._jac_map_callback, 1
            )
        except Exception:
            self.get_logger().warning("Failed to subscribe to /dls2/jacobian_column_map")

        if USE_SCHEDULER:
            self.timer = self.create_timer(1.0 / SCHEDULER_FREQ, self.compute_control_callback)
            self.get_logger().info(f"Scheduler enabled at {SCHEDULER_FREQ:.1f} Hz")
        else:
            self.get_logger().info("Processing messages as they arrive (no fixed scheduler)")

        # --- State Variables ---
        self.first_message_base_arrived = False
        self.first_message_joints_arrived = False
        self.joint_mapping_valid = False
        self.received_joint_name_list = False
        self.loop_time = 1.0 / SCHEDULER_FREQ if USE_FIXED_LOOP_TIME else 0.004 # Default guess
        self.latest_state_timestamp = None
        self.last_state_timestamp = None
        self.last_valid_simulation_dt = self.loop_time
        self.last_start_time = None
        self.last_mpc_loop_time = 0.0
        self.last_mpc_time = time.time()

        # Base State
        self.position = np.zeros(3)
        self.orientation = np.zeros(4)
        self.linear_velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        self.stance_status = np.zeros(4)

        # Blind State
        self.dof_offset = 6 # 6 base DOFs
        self.num_joints = 12
        self.num_dofs = self.num_joints + self.dof_offset # Total 18 DOFs

        self.joint_positions = np.zeros(self.num_joints)
        self.joint_velocities = np.zeros(self.num_joints)
        self.feet_contact = np.zeros(4)
        self.feet_positions = np.zeros(self.num_joints)
        self.feet_velocities = np.zeros(self.num_joints)
        self.hip_positions = np.zeros(self.num_joints)
        self.mass_matrix = np.zeros((self.num_joints, self.num_joints))
        self.bias_forces = np.zeros(self.num_joints)
        # Jacobians are now (4 legs, 3D, 18 DOFs)
        self.feet_jacobians = np.zeros((4, 3, self.num_dofs))
        self.feet_jacobian_dots = np.zeros((4, 3, self.num_dofs))
        self.base_inertia = np.diag([1.0]*6).flatten()

        # Keep gains for trajectory publisher
        self.impedence_joint_position_gain = np.ones(12) * cfg.simulation_params['impedence_joint_position_gain']
        self.impedence_joint_velocity_gain = np.ones(12) * cfg.simulation_params['impedence_joint_velocity_gain']

        # Joystick References
        self.ref_base_lin_vel_H = np.zeros(3)
        self.ref_base_ang_yaw_dot = 0.0

        # Joint Mapping
        self.expected_joint_names = EXPECTED_JOINT_NAMES
        self.joint_name_map = None

        # --- Hardcoded Indices ---
        self.legs_qpos_idx = LegsAttr(
            FL=np.array([7, 8, 9]), FR=np.array([10, 11, 12]),
            RL=np.array([13, 14, 15]), RR=np.array([16, 17, 18])
        )
        self.legs_qvel_idx = LegsAttr(
            FL=np.array([6, 7, 8]), FR=np.array([9, 10, 11]),
            RL=np.array([12, 13, 14]), RR=np.array([15, 16, 17])
        )
        self.legs_order = ["FL", "FR", "RL", "RR"]

        # --- Torque Limits ---
        # Per-joint limits: hip & thigh have lower max, calf is stronger
        tau_soft_limits_scalar = 0.9
        # Desired absolute maxima (Nm) per joint [hip, thigh, calf]
        per_joint_max = np.array([23.7, 23.7, 45.43], dtype=float)
        # Build limits as (3,2) arrays [[-max, max], ...] and apply soft scalar
        def _limits_for_leg():
            # columns: [min, max] per joint -> shape (3,2)
            arr = np.column_stack((-per_joint_max, per_joint_max))
            return arr * tau_soft_limits_scalar

        self.tau_limits = LegsAttr(
            FL=_limits_for_leg(),
            FR=_limits_for_leg(),
            RL=_limits_for_leg(),
            RR=_limits_for_leg()
        )

        # --- Controller Initialization ---
        self.get_logger().info("Initializing MPC and WBC interfaces...")
        zero_feet = LegsAttr(FL=np.zeros(3), FR=np.zeros(3), RL=np.zeros(3), RR=np.zeros(3))
        self.wb_interface = WBInterface(initial_feet_pos=zero_feet, legs_order=self.legs_order)
        self.srbd_controller_interface = SRBDControllerInterface()
        self.get_logger().info("Controller interfaces initialized.")

        # Shared variables between MPC and WBC
        self.nmpc_GRFs = copy.deepcopy(zero_feet)
        self.nmpc_footholds = copy.deepcopy(zero_feet)
        self.nmpc_joints_pos = copy.deepcopy(zero_feet)
        self.nmpc_joints_vel = copy.deepcopy(zero_feet)
        self.nmpc_joints_acc = copy.deepcopy(zero_feet)
        self.nmpc_predicted_state = np.zeros(12)
        self.best_sample_freq = self.wb_interface.pgg.step_freq

        # Buffers for MPC inputs
        self.state_current = None
        self.ref_state = None
        self.contact_sequence = None
        self.inertia = None
        self.optimize_swing = None

        # Output torque
        self.tau = LegsAttr(*[np.zeros(3) for _ in range(4)])

        # Start in full stance gait
        self.wb_interface.pgg.gait_type = 7
        self.get_logger().info("Defaulting to full stance gait.")

        # --- Start MPC Thread/Process (if enabled) ---
        if USE_THREADED_MPC and USE_PROCESS_MPC:
            raise ValueError("Cannot use both USE_THREADED_MPC and USE_PROCESS_MPC.")

        if USE_THREADED_MPC:
            self.get_logger().info("Starting MPC in a separate thread...")
            thread_mpc = threading.Thread(target=self.compute_mpc_thread_callback)
            thread_mpc.daemon = True
            thread_mpc.start()
        elif USE_PROCESS_MPC:
            self.get_logger().info("Starting MPC in a separate process...")
            self.input_data_process = multiprocessing.Queue(maxsize=1)
            self.output_data_process = multiprocessing.Queue(maxsize=1)
            process_mpc = multiprocessing.Process(
                target=self.compute_mpc_process_callback,
                args=(self.input_data_process, self.output_data_process)
            )
            process_mpc.daemon = True
            process_mpc.start()
        else:
            self.get_logger().info("Running MPC synchronously in the main callback.")

        # --- Keep Interactive Console Thread ---
        self.get_logger().info("Starting interactive console thread...")
        self.console = Console(controller_node=self)
        thread_console = threading.Thread(target=self.console.interactive_command_line)
        thread_console.daemon = True
        thread_console.start()

        self.get_logger().info("Quadruped PyMPC Node initialization complete.")


    # --- MPC Worker Functions (Thread/Process) ---
    def compute_mpc_thread_callback(self):
        """ Runs MPC computations in a dedicated thread. """
        last_mpc_thread_time = time.time()
        self.get_logger().info("MPC thread started.")
        while rclpy.ok():
            if time.time() - last_mpc_thread_time < (1.0 / MPC_FREQ):
                time.sleep(0.001)
                continue

            # Wait for data
            state_current_thread = self.state_current
            ref_state_thread = self.ref_state
            contact_sequence_thread = self.contact_sequence
            inertia_thread = self.inertia
            optimize_swing_thread = self.optimize_swing

            if state_current_thread is None or ref_state_thread is None or contact_sequence_thread is None or inertia_thread is None:
                continue

            try:
                # Compute control (threaded path). Detailed per-input logging was
                # removed to centralize debug output into the main loop.
                res = self.srbd_controller_interface.compute_control(
                    state_current_thread,
                    ref_state_thread,
                    contact_sequence_thread,
                    inertia_thread,
                    self.wb_interface.pgg.phase_signal, # Read current value
                    self.wb_interface.pgg.step_freq,    # Read current value
                    optimize_swing_thread
                )

                if isinstance(res, (tuple, list)) and len(res) == 7:
                    # Update shared attributes (careful with race conditions if read/write overlap)
                    (self.nmpc_GRFs, self.nmpc_footholds, self.nmpc_joints_pos,
                     self.nmpc_joints_vel, self.nmpc_joints_acc, self.best_sample_freq,
                     self.nmpc_predicted_state) = res
                else:
                    self.get_logger().warning(f"Unexpected compute_control return type/length (thread): {type(res)}")

                if cfg.mpc_params['type'] != 'sampling' and cfg.mpc_params['use_RTI']:
                    self.srbd_controller_interface.compute_RTI()

            except Exception as e:
                self.get_logger().error(f"compute_control (thread) raised exception: {e}")

            last_mpc_thread_time = time.time()


    def compute_mpc_process_callback(self, input_queue, output_queue):
        """ Runs MPC computations in a dedicated process. """
        try:
            pid = os.getpid()
            os.nice(-20)
            affinity_mask = {6, 7}
            os.sched_setaffinity(pid, affinity_mask)
        except Exception as e:
            print(f"MPC process: Warning - Failed to set priority/affinity: {e}")

        srbd_interface = SRBDControllerInterface()
        last_log_time = 0.0 # For throttling print

        while True:
            try:
                data = input_queue.get(block=True, timeout=0.1)
            except multiprocessing.queues.Empty:
                continue

            (state_current, ref_state, contact_sequence, inertia,
             optimize_swing, phase_signal, step_freq) = data

            start_compute_time = time.time()
            try:
                # Detailed per-input logging in the MPC process removed.
                # Compute control
                res = srbd_interface.compute_control(
                    state_current, ref_state, contact_sequence, inertia,
                    phase_signal, step_freq, optimize_swing
                )

                if isinstance(res, (tuple, list)) and len(res) == 7:
                    compute_duration = time.time() - start_compute_time
                    output_data = list(res) + [compute_duration]

                    # Clear queue before putting (ensure latest result)
                    while not output_queue.empty():
                        try:
                            output_queue.get_nowait()
                        except multiprocessing.queues.Empty:
                            break
                    output_queue.put_nowait(output_data)

                    if cfg.mpc_params['type'] != 'sampling' and cfg.mpc_params['use_RTI']:
                        srbd_interface.compute_RTI()
                else:
                    print(f"MPC Process: Unexpected compute_control return type/length: {type(res)}")

            except Exception as e:
                print(f"MPC Process: compute_control raised exception: {e}")


    # --- ROS Callback Functions ---
    def get_base_state_callback(self, msg: BaseStateMsg):
        """ Updates base state from simulator message. """
        self.latest_state_timestamp = float(msg.timestamp)
        pos = np.array(msg.position)
        lin_vel = np.array(msg.linear_velocity)

        if USE_SMOOTH_HEIGHT:
            self.position[2] = 0.5 * self.position[2] + 0.5 * pos[2]
            self.position[0:2] = pos[0:2]
        else:
            self.position = pos

        if USE_SMOOTH_VELOCITY:
            self.linear_velocity = 0.5 * self.linear_velocity + 0.5 * lin_vel
        else:
            self.linear_velocity = lin_vel

        self.orientation = np.array(msg.orientation)
        self.angular_velocity = np.array(msg.angular_velocity)
        self.stance_status = np.array(msg.stance_status)
        self.first_message_base_arrived = True


    def _set_joint_mapping(self, sim_joint_names: list):
        """ Computes mapping from controller joint order to simulator joint order. """
        self.get_logger().info(f"Received simulator joint names: {sim_joint_names}")
        try:
            sim_names = list(sim_joint_names)
            map_idx = [sim_names.index(expected_name) for expected_name in self.expected_joint_names]
            self.joint_name_map = np.array(map_idx, dtype=int)
            self.joint_mapping_valid = True
            self.received_joint_name_list = True
            self.get_logger().info(f"Joint-name mapping established. Controller -> Sim indices: {self.joint_name_map.tolist()}")
        except ValueError:
            missing = [name for name in self.expected_joint_names if name not in sim_names]
            self.get_logger().error(f"Simulator joint names missing expected joints: {missing}. Sim has: {sim_names[:20]}...")
            self.joint_mapping_valid = False
            self.joint_name_map = None
        except Exception as e:
            self.get_logger().error(f"_set_joint_mapping failed unexpectedly: {e}")
            self.joint_mapping_valid = False
            self.joint_name_map = None



    def get_blind_state_callback(self, msg: BlindStateMsg):
        """ Updates joint states and dynamics info from simulator message. """
        self.latest_state_timestamp = float(msg.timestamp)
        sim_joint_names = getattr(msg, 'joints_name', None)
        if sim_joint_names and not self.received_joint_name_list:
            # We still run this to *check* the mapping, but we won't use it on the data.
            self._set_joint_mapping(list(sim_joint_names)) 
            self.first_message_joints_arrived = True # Mark as ready

        # --- Read Raw Data ---
        # This data is already in CONTROLLER ORDER from RosMpcController
        self.joint_positions = np.array(getattr(msg, 'joints_position', np.zeros(12)))
        self.joint_velocities = np.array(getattr(msg, 'joints_velocity', np.zeros(12)))
        self.feet_contact = np.array(getattr(msg, 'feet_contact', np.zeros(4)))
        self.feet_positions = np.array(getattr(msg, 'feet_positions', np.zeros(12)))
        self.feet_velocities = np.array(getattr(msg, 'feet_velocities', np.zeros(12)))
        self.hip_positions = np.array(getattr(msg, 'hip_positions', np.zeros(12)))
        self.bias_forces = np.array(getattr(msg, 'bias_forces', np.zeros(12)))

        try:
            raw_mass_matrix_flat = getattr(msg, 'mass_matrix', [])
            self.mass_matrix = np.array(raw_mass_matrix_flat).reshape(12, 12) if len(raw_mass_matrix_flat) == 144 else np.zeros((12, 12))
        except Exception:
            self.mass_matrix = np.zeros((12, 12))

        try:
            base_inertia_flat = getattr(msg, 'base_inertia', [])
            # Validate received inertia: expect length 9 and all finite numbers
            if isinstance(base_inertia_flat, (list, tuple, np.ndarray, array.array)) and len(base_inertia_flat) == 9:
                arr = np.array(base_inertia_flat, dtype=float)
                if np.all(np.isfinite(arr)):
                    self.base_inertia = arr
                else:
                    # keep previous valid value or fallback to identity
                    self.get_logger().warning("Received base_inertia contains non-finite values; keeping previous value or using identity fallback.")
                    if not (hasattr(self, 'base_inertia') and self.base_inertia.size == 9 and np.all(np.isfinite(self.base_inertia))):
                        self.base_inertia = np.eye(3).flatten()
            else:
                # If not provided or wrong size, keep previous or set identity
                if not (hasattr(self, 'base_inertia') and self.base_inertia.size == 9 and np.all(np.isfinite(self.base_inertia))):
                    self.get_logger().warning("base_inertia missing or wrong size in message; using identity fallback.")
                    self.base_inertia = np.eye(3).flatten()
        except Exception as e:
            self.get_logger().error(f"Error processing base_inertia from message: {e}")
            # Ensure a valid fallback inertia
            self.base_inertia = np.eye(3).flatten()

        # --- Jacobians ---
        try:
            raw_jac_flat = getattr(msg, 'feet_jacobians', [])
            expected_jac_len = 4 * 3 * self.num_dofs # 4 * 3 * 18 = 216
            self.feet_jacobians = np.array(raw_jac_flat).reshape(4, 3, self.num_dofs) if len(raw_jac_flat) == expected_jac_len else np.zeros((4, 3, self.num_dofs))
        except Exception:
            self.feet_jacobians = np.zeros((4, 3, self.num_dofs))

        try:
            raw_jac_dot_flat = getattr(msg, 'feet_jacobian_dots', [])
            expected_jac_dot_len = 4 * 3 * self.num_dofs # 4 * 3 * 18 = 216
            self.feet_jacobian_dots = np.array(raw_jac_dot_flat).reshape(4, 3, self.num_dofs) if len(raw_jac_dot_flat) == expected_jac_dot_len else np.zeros((4, 3, self.num_dofs))
        except Exception:
            self.feet_jacobian_dots = np.zeros((4, 3, self.num_dofs))

        # --- No mapping section is needed ---

        # --- Apply DLS Convention Fix (if needed) ---
        if USE_DLS_CONVENTION:
            self.joint_positions[0] *= -1.0
            # ... (rest of DLS logic) ...

        # self.first_message_joints_arrived = True # Moved up
        
        # --- Trigger Control Computation (only when not using scheduler) ---
        if not USE_SCHEDULER:
            self.compute_control_callback()


    def get_joy_callback(self, msg: Joy):
        """ Updates reference velocities from joystick input. """
        try:
            self.ref_base_lin_vel_H[0] = -msg.axes[1] * cfg.joystick_params['vx_scale'] # Forward
            self.ref_base_lin_vel_H[1] = -msg.axes[0] * cfg.joystick_params['vy_scale'] # Left Strafe
            self.ref_base_ang_yaw_dot = -msg.axes[3] * cfg.joystick_params['dyaw_scale'] # Yaw Rate

            if len(msg.buttons) > 8 and msg.buttons[8] == 1:
                self.get_logger().info("Shutdown button pressed on joystick. Exiting.")
                rclpy.shutdown()
        except IndexError:
            self.get_logger().warning("Joystick message axes/buttons out of range.", throttle_duration_sec=5.0)
        except KeyError:
             self.get_logger().warning("Joystick params missing in config.", throttle_duration_sec=5.0)


# --- Main Control Computation ---
    def compute_control_callback(self):
        """ Main loop for computing and publishing control commands. """
        start_time = time.perf_counter()

        if not self.first_message_base_arrived or not self.first_message_joints_arrived:
            return

        # --- Calculate Loop Time ---
        # if USE_FIXED_LOOP_TIME:
        #     simulation_dt = 1.0 / SCHEDULER_FREQ
        # else:
        #     if self.last_start_time is not None:
        #         self.loop_time = start_time - self.last_start_time
        #     else:
        #         self.loop_time = 1.0 / SCHEDULER_FREQ # First loop guess
        #     simulation_dt = self.loop_time
        #     if USE_SATURATED_LOOP_TIME and simulation_dt > MAX_LOOP_TIME_S:
        #         simulation_dt = MAX_LOOP_TIME_S
        if USE_FIXED_LOOP_TIME:
            simulation_dt = 1.0 / SCHEDULER_FREQ
        else:
            if self.latest_state_timestamp is not None and self.last_state_timestamp is not None:
                simulation_dt = self.latest_state_timestamp - self.last_state_timestamp
            elif self.last_start_time is not None:
                simulation_dt = start_time - self.last_start_time
            else:
                simulation_dt = 1.0 / SCHEDULER_FREQ

            if simulation_dt <= 0.0:
                simulation_dt = self.last_valid_simulation_dt

            if USE_SATURATED_LOOP_TIME and simulation_dt > MAX_LOOP_TIME_S:
                simulation_dt = MAX_LOOP_TIME_S

        self.last_start_time = start_time
        self.loop_time = simulation_dt
        self.last_valid_simulation_dt = simulation_dt

        if self.latest_state_timestamp is not None:
            self.last_state_timestamp = self.latest_state_timestamp

        # --- Prepare State for Controller ---
        try:
            #norm = np.linalg.norm(self.orientation)
            # Use received xyzw quat
            #quat = self.orientation / norm if abs(norm - 1.0) > 1e-6 else self.orientation
            quat = self.orientation
            R = Rotation.from_quat(quat).as_matrix() # Expects xyzw
            base_ori_euler_xyz = Rotation.from_quat(quat).as_euler('xyz')
        except ValueError:
             self.get_logger().error(f"Invalid orientation quaternion {self.orientation}. Using identity.", throttle_duration_sec=2.0)
             R = np.identity(3)
             base_ori_euler_xyz = np.zeros(3)
             quat = np.array([0.0, 0.0, 0.0, 1.0]) # xyzw

        # --- ADJUSTMENT: Keep world lin vel, convert ang vel to body ---
        # base_lin_vel_B = R.T @ self.linear_velocity # REMOVED: wb_interface expects world frame
        base_ang_vel_B = R.T @ self.angular_velocity # KEEP: wb_interface expects body frame
        #base_ang_vel_B = self.angular_velocity

        # qpos/qvel construction (uses world velocities)
        qpos = np.concatenate([self.position, quat, self.joint_positions])
        qvel = np.concatenate([self.linear_velocity, base_ang_vel_B, self.joint_velocities])  #qvel needs base ang vel in body frame

        # --- Prepare LegsAttr data ---
        feet_pos = LegsAttr(FL=self.feet_positions[0:3], FR=self.feet_positions[3:6],
                            RL=self.feet_positions[6:9], RR=self.feet_positions[9:12])
        feet_vel = LegsAttr(FL=self.feet_velocities[0:3], FR=self.feet_velocities[3:6],
                            RL=self.feet_velocities[6:9], RR=self.feet_velocities[9:12])
        hip_pos = LegsAttr( FL=self.hip_positions[0:3], FR=self.hip_positions[3:6],
                            RL=self.hip_positions[6:9], RR=self.hip_positions[9:12])
        joints_pos = LegsAttr(FL=self.joint_positions[0:3], FR=self.joint_positions[3:6],
                              RL=self.joint_positions[6:9], RR=self.joint_positions[9:12])

        legs_mass_matrix = LegsAttr(
            FL=self.mass_matrix[0:3, 0:3], FR=self.mass_matrix[3:6, 3:6],
            RL=self.mass_matrix[6:9, 6:9], RR=self.mass_matrix[9:12, 9:12]
        )
        legs_qfrc_bias = LegsAttr(
            FL=self.bias_forces[0:3], FR=self.bias_forces[3:6],
            RL=self.bias_forces[6:9], RR=self.bias_forces[9:12]
        )
        legs_qfrc_passive = LegsAttr(FL=np.zeros(3), FR=np.zeros(3), RL=np.zeros(3), RR=np.zeros(3))

        feet_jac = LegsAttr(FL=self.feet_jacobians[0], FR=self.feet_jacobians[1],
                            RL=self.feet_jacobians[2], RR=self.feet_jacobians[3])
        feet_jac_dot = LegsAttr(FL=self.feet_jacobian_dots[0], FR=self.feet_jacobian_dots[1],
                                RL=self.feet_jacobian_dots[2], RR=self.feet_jacobian_dots[3])

        # --- Reference velocity calculation ---
        yaw = base_ori_euler_xyz[2]
        R_yaw = Rotation.from_euler('z', yaw).as_matrix()
        ref_base_lin_vel_W = R_yaw @ self.ref_base_lin_vel_H
        ref_base_ang_vel_W = np.array([0.0, 0.0, self.ref_base_ang_yaw_dot])

        # # --- ADJUSTMENT: Force use of config inertia, ignore sim inertia ---
        # if hasattr(cfg, 'inertia') and isinstance(cfg.inertia, np.ndarray) and cfg.inertia.shape == (3, 3):
        #     current_inertia = cfg.inertia.flatten() # Use config value
        #     if not np.all(np.isfinite(current_inertia)):
        #          self.get_logger().error("Config inertia (cfg.inertia) contains NaN/Inf! Using identity fallback.")
        #          current_inertia = np.eye(3).flatten()
        # else:
        #     self.get_logger().error("Config inertia (cfg.inertia) is not 3x3! Using identity fallback.")
        #     current_inertia = np.eye(3).flatten()
        # # --- END ADJUSTMENT ---
        if hasattr(self, 'base_inertia') and self.base_inertia.size == 9 and np.all(np.isfinite(self.base_inertia)):
            current_inertia = self.base_inertia.flatten()
        else:
            self.get_logger().error("Base inertia from sim is invalid! Using identity fallback.")
            current_inertia = np.eye(3).flatten()
        
        heightmaps = None # Placeholder
        #print(current_inertia)

        # --- Update State and Reference in WBC Interface ---
        state_current, \
        ref_state, \
        contact_sequence, \
        step_height, \
        optimize_swing = self.wb_interface.update_state_and_reference(
            self.position, self.position, # Use base_pos for com_pos approximation
            self.linear_velocity,         # <-- ADJUSTMENT: Pass WORLD Linear Velocity
            base_ori_euler_xyz,           # Base Orientation (Euler XYZ)
            base_ang_vel_B,               # <-- Pass BODY Angular Velocity
            feet_pos, hip_pos, joints_pos, heightmaps,
            self.legs_order, simulation_dt, ref_base_lin_vel_W, ref_base_ang_vel_W
        )

        # --- Keep Console Adjustments ---
        ref_state["ref_position"][2] += self.console.height_delta
        ref_state["ref_orientation"][1] += self.console.pitch_delta # Pitch adjustment

        # --- Contact sequence processing (remains the same) ---
        try:
            horizon = int(cfg.mpc_params.get('horizon', 12))
            cs = np.asarray(contact_sequence)
            if cs.ndim == 1 and cs.size == 4:
                contact_sequence_mpc = np.tile(cs.reshape(4, 1), (1, horizon)).astype(float)
            elif cs.ndim == 2 and cs.shape[0] == 4:
                pad_len = horizon - cs.shape[1]
                if pad_len > 0:
                    pad = np.ones((4, pad_len), dtype=float)
                    contact_sequence_mpc = np.concatenate([cs, pad], axis=1).astype(float)
                else:
                    contact_sequence_mpc = cs[:, :horizon].astype(float)
            else:
                 contact_sequence_mpc = np.ones((4, horizon), dtype=float)
        except Exception as e:
            self.get_logger().error(f"Error processing contact sequence: {e}. Defaulting to full stance.")
            contact_sequence_mpc = np.ones((4, int(cfg.mpc_params.get('horizon', 12))), dtype=float)

        # --- Trigger or Retrieve MPC Results ---
        if USE_THREADED_MPC:
            self.state_current = state_current
            self.ref_state = ref_state
            self.contact_sequence = contact_sequence_mpc
            self.inertia = current_inertia
            self.optimize_swing = optimize_swing

        elif USE_PROCESS_MPC:
            if not self.input_data_process.full():
                self.input_data_process.put_nowait([
                    state_current, ref_state, contact_sequence_mpc, current_inertia,
                    optimize_swing, self.wb_interface.pgg.phase_signal, self.wb_interface.pgg.step_freq
                ])
            if not self.output_data_process.empty():
                try:
                    data = self.output_data_process.get_nowait()
                    (self.nmpc_GRFs, self.nmpc_footholds, self.nmpc_joints_pos,
                     self.nmpc_joints_vel, self.nmpc_joints_acc, self.best_sample_freq,
                     self.nmpc_predicted_state, self.last_mpc_loop_time) = data
                except multiprocessing.queues.Empty: pass
                except Exception as e:
                     self.get_logger().error(f"Error getting MPC result from process queue: {e}")

        else:  # Synchronous MPC
            if time.time() - self.last_mpc_time > (1.0 / MPC_FREQ):
                mpc_start_time = time.time()
                try:
                    # Compute MPC synchronously without verbose per-input logging.
                    res = self.srbd_controller_interface.compute_control(
                        state_current, ref_state, contact_sequence_mpc, current_inertia,
                        self.wb_interface.pgg.phase_signal, self.wb_interface.pgg.step_freq, optimize_swing
                    )

                    if isinstance(res, (tuple, list)) and len(res) == 7:
                        (self.nmpc_GRFs, self.nmpc_footholds, self.nmpc_joints_pos,
                         self.nmpc_joints_vel, self.nmpc_joints_acc, self.best_sample_freq,
                         self.nmpc_predicted_state) = res
                    elif res is not None:
                        self.get_logger().warning(f"Unexpected compute_control return type/length (sync): {type(res)}")

                    if cfg.mpc_params['type'] != 'sampling' and cfg.mpc_params['use_RTI']:
                        self.srbd_controller_interface.compute_RTI()

                except Exception as e:
                    self.get_logger().error(f"compute_control (sync) raised exception: {e}")
                    import traceback # Import here for clarity
                    traceback.print_exc() # Print full stack trace on error

                self.last_mpc_loop_time = time.time() - mpc_start_time
                self.last_mpc_time = time.time()

        # --- Compute Swing and Stance Torques (WBC) ---
        
        # --- CHANGED: Remove local index calculation ---
        # local_legs_qpos_idx = ... (REMOVED)
        # local_legs_qvel_idx = ... (REMOVED)

        try:
            # Keep pd_target variables
            self.tau, pd_target_joints_pos, pd_target_joints_vel = self.wb_interface.compute_stance_and_swing_torque(
                simulation_dt, qpos, qvel, feet_jac, feet_jac_dot,
                feet_pos, feet_vel, legs_qfrc_passive, legs_qfrc_bias, legs_mass_matrix,
                self.nmpc_GRFs, self.nmpc_footholds,
                self.legs_qpos_idx, self.legs_qvel_idx, self.tau, # <-- CHANGED: Use global indices
                optimize_swing, self.best_sample_freq,
                self.nmpc_joints_pos, self.nmpc_joints_vel, self.nmpc_joints_acc,
                self.nmpc_predicted_state
            )
            for leg in self.legs_order:
                self.tau[leg] = np.asarray(self.tau[leg]).flatten()

        except Exception as e:
            self.get_logger().error(f"compute_stance_and_swing_torque raised exception: {e}")
            import traceback # Import here for clarity
            traceback.print_exc() # Print full stack trace on error
            self.tau = LegsAttr(*[np.zeros(3) for _ in range(4)]) # Fallback
            # Fallback for PD targets too
            pd_target_joints_pos = LegsAttr(*[np.zeros(3) for _ in range(4)])
            pd_target_joints_vel = LegsAttr(*[np.zeros(3) for _ in range(4)])

        # --- Apply Torque Limits ---
        for leg in self.legs_order:
            tau_min = self.tau_limits[leg][:, 0]
            tau_max = self.tau_limits[leg][:, 1]
            current_tau = np.asarray(self.tau[leg]).flatten()
            self.tau[leg] = np.clip(current_tau, tau_min, tau_max)

        # --- Apply DLS Convention Fix (if needed) ---
        if USE_DLS_CONVENTION:
            self.tau.FL[0] *= -1.0
            self.tau.RL[0] *= -1.0
            # Keep PD target flips for trajectory publisher
            pd_target_joints_pos.FL[0] *= -1.0
            pd_target_joints_pos.RL[0] *= -1.0
            pd_target_joints_vel.FL[0] *= -1.0
            pd_target_joints_vel.RL[0] *= -1.0

        # --- Optional centralized debug logging (single block) ---
        if getattr(self, 'ENABLE_DEBUG_LOGGING', False):
            
            logger = self.get_logger()
            logger.info(f"--- [DEBUG LOG] Loop Time: {simulation_dt:.4f} ---")

            # 1. MuJoCo Simulation State (Inputs to controller)
            # logger.info(f"== [SIM STATE] ==")
            # logger.info(f"  Base Pos/Ori (qpos[0:7]): {qpos[0:7]}")
            # logger.info(f"  Joint Pos (qpos[7:]): {qpos[7:]}")
            # logger.info(f"  Base Vel (qvel[0:6]): {qvel[0:6]}")
            # logger.info(f"  Joint Vel (qvel[6:]): {qvel[6:]}")
            # logger.info(f"  current inertia: {current_inertia}")

            # # 2. MuJoCo Derived Dynamics (Showing FL leg as example)
            # logger.info(f"== [DYNAMICS (FL Leg Example)] ==")
            # logger.info(f"  FL Mass Matrix: \n{legs_mass_matrix.FL}")
            # logger.info(f"  FR Mass Matrix: \n{legs_mass_matrix.FR}")
            # logger.info(f"  RL Mass Matrix: \n{legs_mass_matrix.RL}")
            # logger.info(f"  RR Mass Matrix: \n{legs_mass_matrix.RR}")
            logger.info(f"  FL Bias (qfrc_bias): {legs_qfrc_bias.FL.flatten()}")
            logger.info(f"  FR Bias (qfrc_bias): {legs_qfrc_bias.FR.flatten()}")
            logger.info(f"  RL Bias (qfrc_bias): {legs_qfrc_bias.RL.flatten()}")
            logger.info(f"  RR Bias (qfrc_bias): {legs_qfrc_bias.RR.flatten()}")
            # logger.info(f"  FL Passive (qfrc_passive): {legs_qfrc_passive.FL.flatten()}")
            # logger.info(f"  FL Feet Jacobian: \n{feet_jac.FL}")
            # logger.info(f"  FL Feet Jacobian Dot: \n{feet_jac_dot.FL}")
            # logger.info(f"  FR Bias (qfrc_bias): {legs_qfrc_bias.FR.flatten()}")
            # logger.info(f"  FR Passive (qfrc_passive): {legs_qfrc_passive.FR.flatten()}")
            # logger.info(f"  FR Feet Jacobian: \n{feet_jac.FR}")
            # logger.info(f"  FR Feet Jacobian Dot: \n{feet_jac_dot.FR}")
            # logger.info(f"  RL Bias (qfrc_bias): {legs_qfrc_bias.RL.flatten()}")
            # logger.info(f"  RL Passive (qfrc_passive): {legs_qfrc_passive.RL.flatten()}")
            # logger.info(f"  RL Feet Jacobian: \n{feet_jac.RL}")
            # logger.info(f"  RL Feet Jacobian Dot: \n{feet_jac_dot.RL}")
            # logger.info(f"  RR Bias (qfrc_bias): {legs_qfrc_bias.RR.flatten()}")
            # logger.info(f"  RR Passive (qfrc_passive): {legs_qfrc_passive.RR.flatten()}")
            # logger.info(f"  RR Feet Jacobian: \n{feet_jac.RR}")
            # logger.info(f"  RR Feet Jacobian Dot: \n{feet_jac_dot.RR}")

            # # 3. MPC Controller Values (Inputs to WBC)
            # logger.info(f"== [MPC OUTPUTS] ==")
            # logger.info(f"  Contact Sequence: {contact_sequence[0, :]}")
            # logger.info(f"  NMPC GRFs (FL/FR/RL/RR):")
            # logger.info(f"    FL: {self.nmpc_GRFs.FL} | FR: {self.nmpc_GRFs.FR}")
            # logger.info(f"    RL: {self.nmpc_GRFs.RL} | RR: {self.nmpc_GRFs.RR}")
            # logger.info(f"  NMPC Footholds (FL/FR/RL/RR):")
            # logger.info(f"    FL: {self.nmpc_footholds.FL} | FR: {self.nmpc_footholds.FR}")
            # logger.info(f"    RL: {self.nmpc_footholds.RL} | RR: {self.nmpc_footholds.RR}")
            # logger.info(f"  NMPC Predicted State: {self.nmpc_predicted_state}")

            # # 4. WBC Controller Values (Final Outputs)
            # logger.info(f"== [WBC OUTPUTS] ==")
            # logger.info(f"  Final Torques (FL/FR/RL/RR):")
            # logger.info(f"    FL: {self.tau.FL.flatten()} | FR: {self.tau.FR.flatten()}")
            # logger.info(f"    RL: {self.tau.RL.flatten()} | RR: {self.tau.RR.flatten()}")
            # logger.info(f"  PD Target Pos (FL/FR/RL/RR):")
            # logger.info(f"    FL: {pd_target_joints_pos.FL.flatten()} | FR: {pd_target_joints_pos.FR.flatten()}")
            # logger.info(f"    RL: {pd_target_joints_pos.RL.flatten()} | RR: {pd_target_joints_pos.RR.flatten()}")
            # logger.info(f"  PD Target Vel (FL/FR/RL/RR):")
            # logger.info(f"    FL: {pd_target_joints_vel.FL.flatten()} | FR: {pd_target_joints_vel.FR.flatten()}")
            # logger.info(f"    RL: {pd_target_joints_vel.RL.flatten()} | RR: {pd_target_joints_vel.RR.flatten()}")
            # logger.info(f"--- [END DEBUG LOG] ---")
        # Limit tau between tau_limits
        for leg in ["FL", "FR", "RL", "RR"]:
            tau_min, tau_max = self.tau_limits[leg][:, 0], self.tau_limits[leg][:, 1]
            self.tau[leg] = np.clip(self.tau[leg], tau_min, tau_max)




        # --- Publish Control Signals ---
        control_signal_msg = ControlSignalMsg()
        control_signal_msg.torques = np.concatenate(self.tau.to_list()).flatten().tolist()
        control_signal_msg.timestamp = float(self.latest_state_timestamp) if self.latest_state_timestamp is not None else 0.0
        self.publisher_control_signal.publish(control_signal_msg)

        # --- Keep Trajectory Generator Publishing ---
        trajectory_generator_msg = TrajectoryGeneratorMsg()
        jp = np.concatenate(pd_target_joints_pos.to_list()).flatten()
        jv = np.concatenate(pd_target_joints_vel.to_list()).flatten()
        trajectory_generator_msg.joints_position = jp.tolist()
        trajectory_generator_msg.joints_velocity = jv.tolist()

        try:
            current_stance = [bool(contact_sequence_mpc[i, 0]) for i in range(4)]
        except IndexError:
             current_stance = [True] * 4 # Fallback
        trajectory_generator_msg.stance_legs = current_stance

        trajectory_generator_msg.kp = self.impedence_joint_position_gain.tolist()
        trajectory_generator_msg.kd = self.impedence_joint_velocity_gain.tolist()
        trajectory_generator_msg.timestamp = float(self.last_mpc_loop_time) # Send MPC computation time
        self.publisher_trajectory_generator.publish(trajectory_generator_msg)

    # --- Main Execution ---
def main(args=None):
    print('Starting Quadruped-PyMPC ROS 2 interface node...')
    rclpy.init(args=args)
    controller_node = None
    try:
        controller_node = Quadruped_PyMPC_Node()
        rclpy.spin(controller_node)
    except KeyboardInterrupt:
        print("Ctrl-C detected, shutting down.")
    except Exception as e:
        print(f"An error occurred during execution: {e}")
    finally:
        if controller_node is not None and isinstance(controller_node, Node):
            controller_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("ROS 2 node shut down.")


if __name__ == '__main__':
    main()
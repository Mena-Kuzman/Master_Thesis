import rclpy
from rclpy.node import Node
from dataclasses import dataclass
import numpy as np
import time
from typing import Optional, Tuple, List, Dict, Any

# Import the required ROS messages
# Ensure dls2_msgs is in your Python path (e.g., by sourcing your ROS workspace)
from dls2_msgs.msg import BaseStateMsg, BlindStateMsg, ControlSignalMsg, TrajectoryGeneratorMsg

# class RosHandler:
#     @staticmethod
#     def rclpy_init():
#         """Initializes the rclpy library if not already initialized."""
#         if not rclpy.ok():
#             rclpy.init()
    


class RosMpcController(Node):
    """
    A minimal ROS2 wrapper to communicate with the quadruped_pympc_node.

    This class is intended to be used from within the Isaac Sim script.
    It encapsulates the logic for:
    1. Publishing BaseStateMsg and BlindStateMsg.
    2. Subscribing to ControlSignalMsg.
    3. Providing a blocking 'get_action' method that sends state and
       waits for the computed torque command.

    NOTE: This class assumes rclpy.init() has been called *before* #todo do this in this init or write a method 
    it is instantiated. The main simulation loop is responsible for
    calling rclpy.spin_once() or similar if 'get_action' is not used.
    """

    @dataclass
    class Params:
        """Parameters for the RosMpcController."""
        # --- Topic Names ---
        base_state_topic: str = "/dls2/base_state"
        blind_state_topic: str = "/dls2/blind_state"
        control_signal_topic: str = "/dls2/quadruped_pympc_torques"
        trajectory_generator_topic: str = "/dls2/trajectory_generator"
        #control_frequency_hz: float = 10

        # --- Robot/Controller Config ---
        # List of joint names in the *controller's* expected order.
        # This is used to populate the 'joints_name' field in BlindStateMsg.
        controller_joint_names: List[str] = (
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        )
        
        # Timeout in seconds for the blocking get_action call
        action_timeout_s: float = 0.5 # 500ms
        
        # Set to True to apply DLS sign convention flips before publishing
        use_dls_convention: bool = False
        

    def __init__(self, params: Params):
        """
        Initializes the ROS2 node, publishers, and subscribers.
        Assumes rclpy.init() has already been called.
        """
        super().__init__('ros_mpc_controller_wrapper')
        self.params = params
        self.ref_base_lin_vel_w = np.zeros(3, dtype=np.float32)
        self.ref_base_ang_vel_w = np.zeros(3, dtype=np.float32)

        # --- ROS Communication ---
        self.pub_base_state = self.create_publisher(
            BaseStateMsg, self.params.base_state_topic, 1
        )
        self.pub_blind_state = self.create_publisher(
            BlindStateMsg, self.params.blind_state_topic, 1
        )
        self.sub_control_signal = self.create_subscription(
            ControlSignalMsg,
            self.params.control_signal_topic,
            self._control_callback,
            1
        )
        self.sub_trajectory_generator = self.create_subscription(
            TrajectoryGeneratorMsg,
            self.params.trajectory_generator_topic,
            self._trajectory_callback,
            1
        )
        
        self.all_topics = {
            "publishing": [
                self.pub_base_state.topic_name,
                self.pub_blind_state.topic_name
            ],
            "subscribing": [
                self.sub_control_signal.topic_name,
                self.sub_trajectory_generator.topic_name
            ]
        }

        # --- State for Blocking Call ---
        self.latest_action_msg: Optional[ControlSignalMsg] = None
        self.action_received: bool = False
        self.action_msg_counter: int = 0

        # Latest trajectory generator PD targets (for impedance overlay in Isaac)
        self.latest_pd_joints_pos = None
        self.latest_pd_joints_vel = None
        self.latest_pd_kp = None
        self.latest_pd_kd = None

        #self.timer = self.create_timer(0.002, self.publish_state)


        self.get_logger().info(
            f"RosMpcController initialized. Waiting for action on: "
            f"'{self.params.control_signal_topic}'"
        )

    def _control_callback(self, msg: ControlSignalMsg):
        """Internal callback to store the latest action message."""
        self.latest_action_msg = msg
        self.action_received = True
        self.action_msg_counter += 1

    def _trajectory_callback(self, msg: TrajectoryGeneratorMsg):
        """Store latest PD targets for optional impedance overlay."""
        try:
            q_des = np.array(msg.joints_position, dtype=np.float32)
            qd_des = np.array(msg.joints_velocity, dtype=np.float32)
            kp = np.array(msg.kp, dtype=np.float32)
            kd = np.array(msg.kd, dtype=np.float32)
        except Exception as exc:
            self.get_logger().warning(f"Failed to parse TrajectoryGeneratorMsg: {exc}")
            return

        if q_des.size != 12 or qd_des.size != 12 or kp.size != 12 or kd.size != 12:
            self.get_logger().warning("TrajectoryGeneratorMsg sizes do not match 12-DoF. Ignoring.")
            return

        self.latest_pd_joints_pos = q_des
        self.latest_pd_joints_vel = qd_des
        self.latest_pd_kp = kp
        self.latest_pd_kd = kd

    def publish_state(self, state_data: Dict[str, np.ndarray]):
        """
        Constructs and publishes the BaseStateMsg and BlindStateMsg.
        This is a non-blocking call.
        """
        try:
            # --- 1. Construct BaseStateMsg ---
            base_state_msg = BaseStateMsg()
            sim_time = state_data.get('timestamp', 0.0)
            base_state_msg.timestamp = float(sim_time)
            
            root_state = state_data['root_state_w']
            base_state_msg.position = root_state[0:3].tolist()
            # Convert wxyz (Isaac) to xyzw (ROS)
            base_state_msg.orientation = np.roll(root_state[3:7], -1).tolist() 
            base_state_msg.linear_velocity = root_state[7:10].tolist()
            base_state_msg.angular_velocity = root_state[10:13].tolist()
            base_state_msg.stance_status = state_data['contact_list'].tolist()
            base_state_msg.frame_id = "world"
            base_state_msg.robot_name = "unitree_go2"

            # --- 2. Construct BlindStateMsg ---
            blind_state_msg = BlindStateMsg()
            blind_state_msg.timestamp = base_state_msg.timestamp
            blind_state_msg.joints_name = self.params.controller_joint_names

            # Get state data
            pos_pub = state_data['joint_pos_ctrl'].copy()
            vel_pub = state_data['joint_vel_ctrl'].copy()
            bias_pub = state_data['bias_ctrl'].copy()

            # Apply DLS convention if needed
            if self.params.use_dls_convention:
                 pos_pub[0] *= -1.0; pos_pub[6] *= -1.0
                 vel_pub[0] *= -1.0; vel_pub[6] *= -1.0
                 bias_pub[0] *= -1.0; bias_pub[6] *= -1.0

            blind_state_msg.joints_position = pos_pub.tolist()
            blind_state_msg.joints_velocity = vel_pub.tolist()
            blind_state_msg.feet_contact = state_data['contact_list'].tolist()
            blind_state_msg.feet_positions = state_data['feet_pos_w'].flatten().tolist()
            blind_state_msg.feet_velocities = state_data['feet_vel_w'].flatten().tolist()
            blind_state_msg.hip_positions = state_data['hip_pos_w'].flatten().tolist()
            blind_state_msg.mass_matrix = state_data['mass_ctrl'].flatten().tolist()
            blind_state_msg.base_inertia = state_data['base_inertia_3x3'].flatten().tolist()
            blind_state_msg.bias_forces = bias_pub.tolist()
            blind_state_msg.feet_jacobians = state_data['feet_jac'].flatten().tolist()
            blind_state_msg.feet_jacobian_dots = state_data['feet_jac_dot'].flatten().tolist()
            blind_state_msg.frame_id = "world"
            blind_state_msg.robot_name = "unitree_go2" 

        except KeyError as e:
            self.get_logger().error(f"Missing key in state_data: {e}. Cannot send state.")
            return
        except Exception as e:
            self.get_logger().error(f"Error constructing state messages: {e}")
            return

        # --- 3. Publish Messages ---
        self.pub_base_state.publish(base_state_msg)
        self.pub_blind_state.publish(blind_state_msg)
        # This method now returns immediately.

    def set_reference_velocity(self, lin_vel_world: np.ndarray, ang_vel_world: np.ndarray) -> None:
        """Store the latest reference velocities (not published by default)."""
        self.ref_base_lin_vel_w = np.asarray(lin_vel_world, dtype=np.float32)
        self.ref_base_ang_vel_w = np.asarray(ang_vel_world, dtype=np.float32)

    def get_pd_targets(self):
        """Return latest PD targets for impedance overlay (or None if unavailable)."""
        if self.latest_pd_joints_pos is None:
            return None
        return (
            self.latest_pd_joints_pos,
            self.latest_pd_joints_vel,
            self.latest_pd_kp,
            self.latest_pd_kd,
        )
        
    def get_action(self, target_timestamp: Optional[float] = None, timestamp_tolerance: float = 1e-4) -> Optional[np.ndarray]:

        start_time = time.perf_counter()
        start_counter = self.action_msg_counter
        # while True:
        #     # Spin to process callbacks (updates self.latest_action_msg)
        #     rclpy.spin_once(self, timeout_sec=0.001)

        #     # Check for timeout
        #     elapsed = time.perf_counter() - start_wait
        #     if elapsed > self.params.action_timeout_s:
        #         # self.get_logger().warn(f"Timeout waiting for timestamp {target_timestamp:.4f}")
        #         return None

        #     # Check if we have a message
        #     if self.latest_action_msg is not None:
        #         msg_time = self.latest_action_msg.timestamp
                
        #         # Check synchronization (tolerance of 0.0001 seconds)
        #         if abs(msg_time - target_timestamp) < 1e-4:
        #             try:
        #                 torques = np.array(self.latest_action_msg.torques, dtype=np.float32)
                        
        #                 # Apply DLS convention if needed
        #                 if self.params.use_dls_convention:
        #                     torques[0] *= -1.0
        #                     torques[6] *= -1.0
                            
        #                 # We found our match! Return immediately.
        #                 return torques
        #             except Exception as e:
        #                 self.get_logger().error(f"Error parsing torques: {e}")
        #                 return None
        #         else:
        #             # Message exists, but timestamps don't match.
        #             # If msg_time < target_time: It's an old message, ignore and keep waiting.
        #             # If msg_time > target_time: We somehow missed a frame? (Rare in sync mode)
        #             pass
        while True:
            # Manually spin the node to process callbacks
            rclpy.spin_once(self, timeout_sec=0.002)

            elapsed_s = time.perf_counter() - start_time
            if elapsed_s > self.params.action_timeout_s:
                if target_timestamp is None:
                    raise TimeoutError(f"No action received from MPC in {elapsed_s*1000.0:.1f} ms.")
                raise TimeoutError(
                    f"No synchronized action received for timestamp {target_timestamp:.6f} "
                    f"within {elapsed_s*1000.0:.1f} ms."
                )

            # Wait for a new message after this call started
            if self.action_msg_counter <= start_counter:
                continue

            msg = self.latest_action_msg
            if msg is None:
                continue

            if target_timestamp is not None and abs(msg.timestamp - target_timestamp) > timestamp_tolerance:
                continue

            try:
                torques = np.array(msg.torques, dtype=np.float32)
            except Exception as e:
                self.get_logger().error(f"Failed to convert received torques to numpy array: {e}")
                return None

            self.action_received = False
            return torques
    

    # def get_action(self) -> Optional[np.ndarray]:
    #     """
    #     Non-blocking check for the latest action.
    #     k
    #     Returns:
    #         - (12,) np.ndarray of torques if a *new* message has arrived
    #           since the last time this function was called.
    #         - None if no new message has been received.
    #     """
    #     start_time = time.perf_counter()
    #     self.action_received = False
    #     self.latest_action_msg = None

    #     while not self.action_received:
    #         # Manually spin the node to process callbacks
    #         rclpy.spin_once(self, timeout_sec=0.001) # 1ms timeout

    #         elapsed_s = time.perf_counter() - start_time
    #         if elapsed_s > self.params.action_timeout_s:
    #             # Timeout -> no action available
    #             # self.get_logger().warning(
    #             #     f"Timeout: No action received from MPC in {elapsed_s*1000.0:.1f} ms."
    #             # )
    #             return None

    #     # A new message exists!
    #     # end_time = time.perf_counter()
    #     # response_time_ms = (end_time - start_time) * 1000.0

    #     # if self.latest_action_msg is None:
    #     #     self.get_logger().error("Action received flag was set, but message is None.")
    #     #     return None

    #     # Convert torques list -> numpy array and store response time for diagnostics
    #     try:
    #         torques = np.array(self.latest_action_msg.torques, dtype=np.float32)
    #     except Exception as e:
    #         self.get_logger().error(f"Failed to convert received torques to numpy array: {e}")
    #         return None
        
    #     #self.last_response_time_ms = response_time_ms
    #     #expected_n = len(self.params.controller_joint_names)

    #     # Diagnostic: unexpected size
    #     # if torques.size != expected_n:
    #     #     self.get_logger().warning(
    #     #         f"Received torques vector of size {torques.size}; expected {expected_n}. "
    #     #         f"Sample: {torques.flatten()[:min(8, torques.size)].tolist()}"
    #     #     )
    #     #     # Also keep last raw for debugging
    #     #     self.last_received_torques_raw = list(self.latest_action_msg.torques)

    #     # # Apply DLS convention fix to *received* torques if needed
    #     # if self.params.use_dls_convention:
    #     #     torques[0] *= -1.0
    #     #     torques[6] *= -1.0

    #     return torques

    def probe_input_output(self):
        """Prints all topics this node publishes to and subscribes from."""
        print("--- RosMpcController Topic Probe ---")
        print("\nPublishing to:")
        for topic in self.all_topics["publishing"]:
            print(f"  - {topic} (Type: {self.pub_base_state.msg_type.__name__ if 'base' in topic else self.pub_blind_state.msg_type.__name__})")
            
        print("\nSubscribing to:")
        for topic in self.all_topics["subscribing"]:
            print(f"  - {topic} (Type: {self.sub_control_signal.msg_type.__name__})")
        print("--------------------------------------")

    def shutdown(self):
        """Cleans up the node."""
        self.get_logger().info("Shutting down RosMpcController.")
        self.destroy_node()
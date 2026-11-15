import numpy as np
import time
from dataclasses import dataclass, field
from typing import Optional, List

# @dataclass
# class StandingControllerParams:
#     """Parameters for the StandingController."""
#     # The 12-element array of target joint angles for the standing pose.
#     target_joint_pos: np.ndarray = field(default_factory=lambda: np.array([
#         0.1, 0.8, -1.5,  # FL
#        -0.1, 0.8, -1.5,  # FR
#         0.1, 1.0, -1.5,  # RL
#        -0.1, 1.0, -1.5   # RR
#     ], dtype=np.float32))
    
#     # Proportional gain for all joints.
#     kp: float = 50.0
    
#     # Derivative gain for all joints.
#     kd: float = 5.0
    
#     # If True, applies the sign flip to the first and seventh torque command.
#     use_dls_convention: bool = False


class StandingController:
    """
    A simple, non-ROS PD controller to keep the robot in a standing position.
    
    This class implements the 'INIT' state logic from the original
    isaac sim script.
    """
    @dataclass
    class Params:
        """Parameters for the StandingController."""
        #The 12-element array of target joint angles for the standing pose.
        target_joint_pos: np.ndarray = field(default_factory=lambda: np.array([
            0.1, 0.8, -1.5,  # FL
        -0.1, 0.8, -1.5,  # FR
            0.1, 1.0, -1.5,  # RL
        -0.1, 1.0, -1.5   # RR
        ], dtype=np.float32))

        target_joint_seat: np.ndarray = field(default_factory=lambda: np.array([
            0.04, 1.0, -2.7,  # FL
        -0.04, 1.0, -2.7,  # FR
            0.04, 1.0, -2.7,  # RL
        -0.04, 1.0, -2.7   # RR
        ], dtype=np.float32))
        
        # Proportional gain for all joints.
        kp: float = 50.0
        
        # Derivative gain for all joints.
        kd: float = 1.0
        
        # If True, applies the sign flip to the first and seventh torque command.
        use_dls_convention: bool = False
        # Joint name ordering expected by this controller (controller order)
        controller_joint_names: List[str] = (
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        )

    def __init__(self, params: Params):
        """
        Initializes the controller with a configuration object.

        Args:
            params (StandingControllerParams): A dataclass containing all
                configuration parameters for the controller.
        """
        self.params = params

        if self.params.target_joint_pos.shape[0] != 12:
            print(f"Warning: target_joint_pos has shape {self.params.target_joint_pos.shape}, "
                  "but 12 was expected.")

        print(f"StandingController initialized with target pose:\n"
              f"{self.params.target_joint_pos.tolist()}")

    def get_torques(self, 
                    current_joint_pos: np.ndarray, 
                    current_joint_vel: np.ndarray) -> np.ndarray:
        """
        Calculates the PD control torques to maintain the standing pose.

        Args:
            current_joint_pos (np.ndarray): The robot's current 12
                joint positions (in controller order).
            current_joint_vel (np.ndarray): The robot's current 12
                joint velocities (in controller order).

        Returns:
            np.ndarray: The 12 computed torque commands.
        """
        
        # Calculate position and velocity errors
        position_error = self.params.target_joint_pos - current_joint_pos
        
        # Velocity target is zero
        velocity_error = 0.0 - current_joint_vel 
        
        # Calculate PD torques (kp*(target-pos) - kd*vel)
        torques = (self.params.kp * position_error) + (self.params.kd * velocity_error)

        # Match the INIT-state controller: do not strongly clip PD torques here.
        # Keep a very large cap as a safeguard, but allow the PD output through.
        MAX_EFFORT = 20
        torques = np.clip(torques, -MAX_EFFORT, MAX_EFFORT)

        # Apply DLS convention if needed (flip hip joints 0 and 6)
        if self.params.use_dls_convention:
            torques[0] *= -1.0
            torques[6] *= -1.0
            
        return torques

    def is_running(self) -> bool:
        """
        A helper method for compatibility with sim loops.
        This controller always "runs".
        """
        return True

    def shutdown(self):
        """
        A placeholder for compatibility. Does nothing.
        """
        print("StandingController shutting down.")
        pass

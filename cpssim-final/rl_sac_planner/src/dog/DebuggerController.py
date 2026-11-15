import numpy as np
import sys
import time
from dataclasses import dataclass
from pynput import keyboard
from typing import List, Optional

class DebuggerController:
    """
    A non-ROS class that provides keyboard-based torque control.
    
    It runs a keyboard listener in a separate thread that updates
    an internal torque vector. The main simulation loop can poll
    this vector using the `get_torques()` method.

    Controls:
    - Left/Right Arrows: Select joint
    - Up/Down Arrows:    Increase/Decrease torque
    - 'z':               Zero selected joint torque
    - 'r':               Reset all torques to zero
    - 'q':               Quit (stops the listener thread)
    """

    @dataclass
    class Params:
        """Parameters for the DebuggerController."""
        # List of joint names in the *controller's* expected order
        controller_joint_names: List[str] = (
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        )
        
        # Amount to change torque by with each key press
        torque_increment: float = 0.5  # Nm

        # Maximum torque (positive or negative)
        max_torque: float = 10.0  # Nm

    def __init__(self, params: Params):
        """
        Initializes the controller state and starts the keyboard listener.
        """
        self.params = params
        self.num_joints = len(self.params.controller_joint_names)
        self.current_joint_index = 0
        self.target_torques = np.zeros(self.num_joints, dtype=np.float32)
        
        # Flag to signal shutdown to the listener thread
        self.running = True

        # --- Keyboard Listener ---
        # pynput listener runs in its own thread
        self.listener = keyboard.Listener(on_press=self._on_key_press)
        self.listener.start()

        self._print_instructions()
        self._print_status()

    def _print_instructions(self):
        """Prints the control instructions to the console."""
        print("--- Keyboard Joint Debugger (Non-ROS) ---")
        print("[Left/Right]: Select Joint")
        print("[Up/Down]:    +/- Torque")
        print("[z]:          Zero Selected Joint Torque")
        print("[r]:          Reset All Torques")
        print("[q]:          Quit Debugger (stops listener)")
        print("-----------------------------------------")

    def _print_status(self):
        """Updates the console with the current state."""
        joint_name = self.params.controller_joint_names[self.current_joint_index]
        torque = self.target_torques[self.current_joint_index]
        
        # Use sys.stdout.write for a single, updating line
        status_msg = f"\r[Joint {self.current_joint_index:02d}] {joint_name:<15}: {torque: 6.2f} Nm"
        sys.stdout.write(status_msg)
        sys.stdout.flush()

    def _on_key_press(self, key):
        """Callback for keyboard press events."""
        
        if not self.running:
            return False # Stop the listener thread

        # Handle special keys (arrows)
        if key == keyboard.Key.up:
            self.target_torques[self.current_joint_index] += self.params.torque_increment
        elif key == keyboard.Key.down:
            self.target_torques[self.current_joint_index] -= self.params.torque_increment
        elif key == keyboard.Key.right:
            self.current_joint_index = (self.current_joint_index + 1) % self.num_joints
        elif key == keyboard.Key.left:
            self.current_joint_index = (self.current_joint_index - 1) % self.num_joints
        
        # Handle character keys
        elif hasattr(key, 'char'):
            if key.char == 'r':
                self.target_torques.fill(0.0)
                sys.stdout.write("\r") # Clear the line
                sys.stdout.flush()
                print("All torques reset to 0.0")
            elif key.char == 'z':
                self.target_torques[self.current_joint_index] = 0.0
            elif key.char == 'q':
                print("\nShutdown key (q) pressed. Stopping listener.")
                self.running = False # Signal shutdown
                return False # Stop the listener

        # Clamp the torque for the current joint
        self.target_torques[self.current_joint_index] = np.clip(
            self.target_torques[self.current_joint_index],
            -self.params.max_torque,
            self.params.max_torque
        )
        
        # Update the console display
        self._print_status()

    def get_torques(self) -> np.ndarray:
        """
        Returns the current target torques.
        This is the main method to be called by your simulation loop.
        """
        # Return a copy to ensure thread safety
        return self.target_torques.copy()
        
    def is_running(self) -> bool:
        """
        Check if the controller is still running (i.e., 'q' not pressed).
        You can use this to terminate your simulation.
        """
        return self.running

    def shutdown(self):
        """
        Gracefully stops the listener thread.
        Call this when your simulation is exiting.
        """
        print("\nShutting down DebuggerController listener...")
        self.running = False
        self.listener.stop()
        # Wait for the listener thread to actually finish
        self.listener.join()
        print("Listener stopped.")



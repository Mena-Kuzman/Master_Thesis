import argparse
import sys, os
# from IsaacLabFromTysrc.dog.SimpleController import SimpleController 
#import carb



############################################################################################

#initialsing path to the src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)



from src.base.baseRunner import BaseRunner
from src.dog.RosMpcController import RosMpcController 
from src.dog.RandomController import RandomController
from src.dog.StandingController import StandingController
from src.dog.DebuggerController import DebuggerController
import torch
import numpy as np
from enum import Enum
import random
import time
from omni.isaac.dynamic_control import _dynamic_control 
from scipy.spatial.transform import Rotation
import rclpy
import pygame
import omni.graph.core as og

class SimulationState(Enum):
    INIT = 0    # Using StandingController
    ACTIVE = 1  # Using RosMpcController

class DogRunner(BaseRunner):
    # custom
    des_pos = torch.zeros(3, device='cpu')
    
    def __init__(self, world, *args, pygame_width=320, pygame_height=240, **kwargs):
        """Initializes the DogRunner and the StandingController."""
        # --- 1. NEW: PYGAME INITIALIZATION ---
        print("[DogRunner] Initializing Pygame window...")
        pygame.init()
        self.pygame_width = pygame_width
        self.pygame_height = pygame_height
        self.screen = pygame.display.set_mode((self.pygame_width, self.pygame_height))
        pygame.display.set_caption("DogRunner State Control")
        self.font = pygame.font.Font(None, 30)
        self.small_font = pygame.font.Font(None, 24)
        # 1. Call super() FIRST. This calls self.stage.world.reset()
        super().__init__(world, *args, **kwargs)
        
        # 2. Store the dog object
        self.dog = self.stage.dog

        # 3. Create controller
        self.standing_controller = StandingController(StandingController.Params())
        self.mpc_controller = RosMpcController(RosMpcController.Params())

        self.state = SimulationState.INIT
        self.last_state = self.state

        self.stage.world.step(render=False) # Step once after reset
        target_pos_ctrl = self.standing_controller.params.target_joint_seat
        dof_names_sim = list(self.dog.articulation.dof_names)
        
        if dof_names_sim:
            controller_joint_names = self.standing_controller.params.controller_joint_names
            sim_map = {name: idx for idx, name in enumerate(dof_names_sim)}
            ctrl_to_sim_indices = [sim_map.get(name, -1) for name in controller_joint_names]
            
            target_pos_sim = np.zeros(len(dof_names_sim))
            
            for i in range(len(controller_joint_names)):
                sim_idx = ctrl_to_sim_indices[i]
                if sim_idx != -1:
                    target_pos_sim[sim_idx] = target_pos_ctrl[i]
            
            self.dog.articulation.set_joint_positions(target_pos_sim)
        else:
            print("[DogRunner] ERROR: Could not get DOF names to set initial pose.")

        self._dc = _dynamic_control.acquire_dynamic_control_interface()
        self.art_handle = self._dc.get_articulation(self.dog.articulation_root)
        self.initial_dof_props = None # Initialize as None

        if self.art_handle == _dynamic_control.INVALID_HANDLE:
             print(f"[DogRunner] ERROR: Could not get articulation handle for '{self.dog.articulation_root}'")
             sys.exit(1)
        else:
            # Get and STORE the original properties
            self.initial_dof_props = self._dc.get_articulation_dof_properties(self.art_handle)
            if self.initial_dof_props is None:
                print("[DogRunner] ERROR: Failed to get initial DOF properties.")
                # Decide how to handle this - maybe exit or use defaults
            else:
                print("[DogRunner] Stored initial DOF properties.")

        time.sleep(0.1)
        
        # 5. Get joint mapping info (uses the lists we just made)
        self.sim_joint_names = dof_names_sim
        self.controller_joint_names = self.standing_controller.params.controller_joint_names
        
        # 6. Create the joint map
        if not self.sim_joint_names:
            print("ERROR: Sim joint names are empty. Cannot create map.")
            # ... (rest of your error handling) ...
            return

        sim_map = {name: idx for idx, name in enumerate(self.sim_joint_names)}
        ctrl_to_sim_indices = []
        for name in self.controller_joint_names:
            if name in sim_map:
                ctrl_to_sim_indices.append(sim_map[name])
            else:
                print(f"Warning: Controller joint '{name}' not found in sim!")
                ctrl_to_sim_indices.append(-1)
        
        self.ctrl_to_sim_idx_np = np.array(ctrl_to_sim_indices, dtype=int)
        self.valid_ctrl_mask = self.ctrl_to_sim_idx_np >= 0
        self.valid_sim_indices = self.ctrl_to_sim_idx_np[self.valid_ctrl_mask]
        self.num_ctrl_joints = len(self.controller_joint_names)

        self.leg_order = ['FL', 'FR', 'RL', 'RR']
        self.body_indices = {}
        self.body_names_list = list(self.dog.articulation.body_names)
        self.body_names_of_interest = []
        
        for leg in self.leg_order:
            foot_name = f'{leg}_foot'
            hip_name = f'{leg}_hip'
            # Find the index of the name in the full list
            self.body_indices[foot_name] = self.body_names_list.index(foot_name)
            self.body_indices[hip_name] = self.body_names_list.index(hip_name)
            # Add the names we care about to our specific list
            self.body_names_of_interest.extend([foot_name, hip_name])
            
        # 9. STORE CONSTANTS
        self.CONTACT_FORCE_THRESHOLD = 1.0 # Newtons
        self.num_sim_dofs = len(self.sim_joint_names)
        self.dof_offset = 6 # Offset for floating base in dynamics matrices
        self.debug_counter = 0 
        
        print("DogRunner initialized with RosMpcController and StandingController fallback.")

        self.active_dof_props = None
        if self.initial_dof_props is not None:
             self.active_dof_props = self.initial_dof_props.copy()
             self.active_dof_props["stiffness"] = 0.0
             self.active_dof_props["damping"] = 0.6
             print(f"[DogRunner] Prepared DOF properties for ACTIVE state (stiffness=0.0, damping={self.active_dof_props['damping'][0]:.2f}).")

        print("DogRunner initialized.")

        self.last_pre_physics_time = time.perf_counter()  #used for calculating jac dots
        #self.mpc_timeout_s = 0.5  # Revert to INIT if no msg in 500ms
        
        self.sim_dt = self.stage.world.get_physics_dt()          #used for calculating jac dots
        self.last_feet_jac = np.zeros((4, 3, self.num_ctrl_joints + self.dof_offset))
        self.feet_jac_dot_ctrl = np.zeros_like(self.last_feet_jac)
        self.physics_step_counter = 0
    
    def _set_dof_properties(self, props):
        """Helper function to set DOF properties."""
        if self.art_handle != _dynamic_control.INVALID_HANDLE and props is not None:
            success = self._dc.set_articulation_dof_properties(self.art_handle, props)

        elif props is None:
             print("[DogRunner] WARNING: Cannot set DOF properties because target properties are None.")

    def pre_physics(self):
        """Gather all state/dynamics data, run state machine, and get torques."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("[DogRunner] Pygame window closed.")
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1 or event.key == pygame.K_i:
                    self.state = SimulationState.INIT
                elif event.key == pygame.K_2 or event.key == pygame.K_a:
                    self.state = SimulationState.ACTIVE

        # Fill the screen with black
        self.screen.fill((0, 0, 0)) # (R, G, B)
        
        # Render text if font is available
        if self.font and self.small_font:
            # Current State Text
            state_color = (0, 255, 0) if self.state == SimulationState.ACTIVE else (255, 255, 0)
            text_str = f"State: {self.state.name}"
            text_surface = self.font.render(text_str, True, state_color)
            self.screen.blit(text_surface, (10, 10))
            
            # Help Text
            help_text1 = "Press '1' or 'i' for INIT (Stand)"
            help_text2 = "Press '2' or 'a' for ACTIVE (MPC)"
            help_surface1 = self.small_font.render(help_text1, True, (200, 200, 200))
            help_surface2 = self.small_font.render(help_text2, True, (200, 200, 200))
            self.screen.blit(help_surface1, (10, 50))
            self.screen.blit(help_surface2, (10, 80))
        
        # Update the display
        pygame.display.flip()
        # --- END PYGAME ---
        
        # --- 2. STATE SWITCH DETECTION & PROPERTY SETTING ---
        if self.state != self.last_state:
            print(f"===================================================")
            print(f"  [DogRunner] STATE SWITCH: {self.last_state.name} -> {self.state.name}")
            print(f"===================================================")
            
            if self.state == SimulationState.INIT:
                # *** SET INITIAL (STANDING) PROPERTIES ***
                print("[DogRunner] Setting DOF properties back to initial values for INIT state.")
                self._set_dof_properties(self.initial_dof_props)
            elif self.state == SimulationState.ACTIVE:
                # *** SET ACTIVE (TORQUE) PROPERTIES ***
                print("[DogRunner] Setting DOF properties for ACTIVE state (Torque Control).")
                self._set_dof_properties(self.active_dof_props)
                
            self.last_state = self.state # Update the last state
        rclpy.spin_once(self.mpc_controller, timeout_sec=0)
        # --- 1. GATHER STATE ---
        root_poses = self.dog.articulation.get_world_poses() 
        #first index is positions in the world frame of the prims. shape is (M, 3). 
        #Second index is quaternion orientations in the world frame of the prims. Quaternion is scalar-first (w, x, y, z). shape is (M, 4).
        #print(root_poses)
        root_lin_vels_tensor = self.dog.articulation.get_linear_velocities()  # in world frame
        root_ang_vels_tensor = self.dog.articulation.get_angular_velocities() # in world frame
        sim_joint_pos_array = self.dog.articulation.get_joint_positions()
        sim_vel_array = self.dog.articulation.get_joint_velocities()
        

        if sim_joint_pos_array is None:
             self.torques_sim = np.zeros(self.num_sim_dofs)
             return

        # --- 2. GET BODY POSES/VELS USING DYNAMIC CONTROL ---
        num_bodies_interest = len(self.body_names_of_interest)
        body_poses_np_pos = np.zeros((num_bodies_interest, 3))
        body_vels_np_lin = np.zeros((num_bodies_interest, 3))
        body_name_to_idx_map = {name: i for i, name in enumerate(self.body_names_of_interest)}

        for i, body_name in enumerate(self.body_names_of_interest):
             body_handle = self._dc.get_articulation_body(self.art_handle, self.body_indices.get(body_name, -1))
             if body_handle != _dynamic_control.INVALID_HANDLE:
                 pose = self._dc.get_rigid_body_pose(body_handle)
                 body_poses_np_pos[i] = pose.p 
                 lin_vel = self._dc.get_rigid_body_linear_velocity(body_handle)
                 body_vels_np_lin[i] = lin_vel

        # --- 3. PROCESS REST OF STATE DATA ---
        base_lin_vel = root_lin_vels_tensor[0]
        base_ang_vel = root_ang_vels_tensor[0]
        # --- END CHANGE ---

        # Construct root state array using unpacked velocities
        sim_root_state_w = np.concatenate([
            root_poses[0][0], # pos
            root_poses[1][0], # quat
            base_lin_vel,                   # lin_vel
            base_ang_vel                    # ang_vel
        ], axis=0)
        
        sim_pos = sim_joint_pos_array[0]
        sim_vel = sim_vel_array[0]
        
        sim_contact_list = np.zeros(4)
        sensor_debug_print = (self.debug_counter % 100 == 1) # Print near the start of each 100 steps

        #if sensor_debug_print: print("--- Contact Sensor Readings ---")

        for i, leg in enumerate(self.leg_order):
            sensor = self.dog.foot_sensors.get(leg)
            contact_made = 0.0 # Default to no contact
            if sensor:
                reading = sensor.get_current_frame()

                if sensor_debug_print: # Print raw reading for this leg
                    force_str = "No 'force' key"
                    if "force" in reading:
                        force_str = f"Force = {np.round(reading['force'], 2)}"
                    #print(f"  {leg}: Reading = {reading.keys()}, {force_str}")

                # Check force condition
                if (
                    "force" in reading
                    and isinstance(reading["force"], (np.ndarray, list, tuple))
                    and len(reading["force"]) >= 3
                ):
                    force_z = reading["force"][2]
                    if force_z > self.CONTACT_FORCE_THRESHOLD:
                        contact_made = 1.0
                    elif sensor_debug_print and force_z > 0:  # Print if force is positive but below threshold
                        # print(f"    -> {leg} Force Z ({force_z:.2f}) is below threshold ({self.CONTACT_FORCE_THRESHOLD})")
                        pass

            sim_contact_list[i] = contact_made # Assign result

        #if sensor_debug_print: print("-----------------------------")
        
        feet_pos_w_list = []
        feet_vel_w_list = []
        hip_pos_w_list = []
        for leg in self.leg_order:
            foot_name = f'{leg}_foot'
            hip_name = f'{leg}_hip'
            foot_idx_local = body_name_to_idx_map[foot_name]
            hip_idx_local = body_name_to_idx_map[hip_name]
            feet_pos_w_list.append(body_poses_np_pos[foot_idx_local])
            feet_vel_w_list.append(body_vels_np_lin[foot_idx_local])
            hip_pos_w_list.append(body_poses_np_pos[hip_idx_local])

        # --- 4. RE-ORDER JOINT DATA (Sim -> Controller) ---
        current_pos_ctrl = np.zeros(self.num_ctrl_joints)
        current_vel_ctrl = np.zeros(self.num_ctrl_joints)
        current_pos_ctrl[self.valid_ctrl_mask] = sim_pos[self.valid_sim_indices]
        current_vel_ctrl[self.valid_ctrl_mask] = sim_vel[self.valid_sim_indices]

        if self.debug_counter == 2: # Print once, right at the start
            #print("\n--- DEBUGGING JOINT MAP (FRAME 2) ---")
            
            sim_names = self.sim_joint_names
            ctrl_names = self.controller_joint_names
            
            #print("SIM_NAME (SIM_IDX) | SIM_POS | --> | CTRL_NAME (CTRL_IDX) | CTRL_POS")
            #print("-" * 70)

            for i in range(self.num_ctrl_joints):
                ctrl_name = ctrl_names[i]
                sim_idx = self.ctrl_to_sim_idx_np[i]
                
                if sim_idx >= 0:
                    sim_name = sim_names[sim_idx]
                    sim_val = sim_pos[sim_idx]
                    ctrl_val = current_pos_ctrl[i]
                    
                    # This line will show the error. 
                    # The 'sim_name' and 'ctrl_name' should be identical.
                    #print(f"{sim_name:<15} ({sim_idx:2}) | {sim_val:8.4f} | --> | {ctrl_name:<15} ({i:2}) | {ctrl_val:8.4f}")
                else:
                    print(f"!!! MISSING !!!     | N/A    | --> | {ctrl_name:<15} ({i:2}) | 0.0000")

            #print("--- END DEBUG ---\n")

        quat_wxyz = sim_root_state_w[3:7]
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        R = Rotation.from_quat(quat_xyzw).as_matrix()    
        # --- 5. GATHER AND PROCESS DYNAMICS ---
        mass_full = self.dog.articulation.get_mass_matrices()[0] # (18, 18)
        coriolis_full = self.dog.articulation.get_coriolis_and_centrifugal_forces()[0] # (12,)
        gravity_full = self.dog.articulation.get_generalized_gravity_forces()[0] # (12,)
        jac_full = self.dog.articulation.get_jacobians()[0] # (num_bodies=19, 6, 18)
        #coriolis_full = coriolis_g_full + gravity_full
        #print(mass_full.shape)
        #print(coriolis_g_full.shape)
        #print(gravity_full.shape)
        #print(jac_full.shape)
        
        
        # --- 6. RE-ORDER DYNAMICS (Sim -> Controller) ---
        ctrl_dof_indices = self.valid_sim_indices + self.dof_offset
        valid_dof_mask = ctrl_dof_indices < mass_full.shape[0]
        ctrl_dof_indices = ctrl_dof_indices[valid_dof_mask]
        ctrl_mask_for_dofs = self.valid_ctrl_mask[valid_dof_mask]

        #mass_ctrl = np.diag(self.static_mass_diag_ctrl_order)
        mass_ctrl = np.zeros((self.num_ctrl_joints, self.num_ctrl_joints))
        if mass_full.shape == (18, 18) and len(ctrl_dof_indices) == self.num_ctrl_joints:
             mass_ctrl[np.ix_(ctrl_mask_for_dofs, ctrl_mask_for_dofs)] = \
                 mass_full[np.ix_(ctrl_dof_indices, ctrl_dof_indices)]
        #else: Handle potential shape mismatch if needed

        bias_ctrl = np.zeros(self.num_ctrl_joints)
        if coriolis_full.shape == (12,):
             bias_ctrl[self.valid_ctrl_mask] = coriolis_full[self.valid_sim_indices]
        # else: Handle potential shape mismatch if needed
        #print(coriolis_g_full)
        #print(gravity_full)
        #print(coriolis_full)
        #print(bias_ctrl)   
        base_inertia_3x3 = np.zeros((3,3))
        if mass_full.shape[0] >= 6:
            base_inertia_3x3 = mass_full[3:6, 3:6]
        
        feet_jac = np.zeros((4, 3, self.num_ctrl_joints + self.dof_offset))

        if jac_full.shape[2] >= 18 and len(ctrl_dof_indices) == self.num_ctrl_joints:
            # These are the column indices for the 12 joints in the controller's (18-wide) matrix
            joint_write_indices = np.where(ctrl_mask_for_dofs)[0] + self.dof_offset

            for i, leg in enumerate(self.leg_order):
                body_idx = self.body_indices[f'{leg}_foot']
                # This is J_w/w (world-in, world-out) from Isaac Sim
                jac_isaac = jac_full[body_idx, 0:3, :] # (3, 18)
                # Create the (3, 18) destination matrix for J_w/b (body-in, world-out)
                jac_mujoco_style = np.zeros_like(jac_isaac)
                # Part 1: Linear Velocity columns (0-2) are the same
                jac_mujoco_style[:, 0:3] = jac_isaac[:, 0:3]               
                # Part 2: Angular Velocity columns (3-5) must be transformed
                # J_w/b (ang) = J_w/w (ang) * R
                jac_mujoco_style[:, 3:6] = jac_isaac[:, 3:6] @ R
                # Part 3: Joint columns (6+) are the same
                jac_mujoco_style[:, 6:] = jac_isaac[:, 6:]
                # --- Now, re-order the columns to match the controller's joint order ---
                jac_ctrl_order = np.zeros((3, self.num_ctrl_joints + self.dof_offset))
                # Copy base DOFs (which are now in the correct MuJoCo style)
                jac_ctrl_order[:, 0:self.dof_offset] = jac_mujoco_style[:, 0:self.dof_offset]
                # Copy re-ordered joint columns
                jac_ctrl_order[:, joint_write_indices] = jac_mujoco_style[:, ctrl_dof_indices]
                feet_jac[i] = jac_ctrl_order
        # else: Handle potential shape mismatch if needed

        if self.debug_counter > 1: # wait for the first frame
            self.feet_jac_dot_ctrl = (feet_jac - self.last_feet_jac) / self.sim_dt
        
        self.last_feet_jac = feet_jac # Store for next frame
        self.feet_jac_dot_ctrl = np.zeros_like(feet_jac)

        # --- 7. BUILD STATE DICT ---
        state_data = {
            #'timestamp': current_sim_time,
            'root_state_w': sim_root_state_w,
            'contact_list': sim_contact_list,
            'joint_pos_ctrl': current_pos_ctrl,
            'joint_vel_ctrl': current_vel_ctrl,
            'feet_pos_w': np.concatenate(feet_pos_w_list), # Needs flatten check
            'feet_vel_w': np.concatenate(feet_vel_w_list), # Needs flatten check
            'hip_pos_w': np.concatenate(hip_pos_w_list),   # Needs flatten check
            'mass_ctrl': mass_ctrl,
            'base_inertia_3x3': base_inertia_3x3,
            'bias_ctrl': bias_ctrl,
            'feet_jac': feet_jac,
            'feet_jac_dot': self.feet_jac_dot_ctrl,
        }
        
        # --- 8. STATE MACHINE AND ACTION ---
        torques_ctrl = 0.0
        last_control_command = None

        try:
            self.mpc_controller.publish_state(state_data)
        except Exception as e: print(f"[DogRunner] Error calling mpc_controller.publish_state: {e}")

        if self.state == SimulationState.INIT:
            torques_ctrl = self.standing_controller.get_torques(current_pos_ctrl, current_vel_ctrl)
            #last_control_command = torques_ctrl
        elif self.state == SimulationState.ACTIVE:
            #if self.physics_step_counter % 2 == 0:
            try:
                torques_ctrl = self.mpc_controller.get_action()
                print(f"[DogRunner] Received torques from MPC controller: {torques_ctrl}")
                last_control_command = torques_ctrl
                if torques_ctrl is None:
                    print("[DogRunner] No new torque command received from ROS MPC controller.")
                    raise ValueError("No new torque command received")
            except Exception as e:
                print(f"[DogRunner] Error calling mpc_controller.get_action: {e}")
            # else:
            #     torques_ctrl = last_control_command
        else:

            print(f"[DogRunner] WARNING: Unknown simulation state '{self.state}'. Using standing controller as fallback.")


        
        # --- 9. MAP ACTION TO SIM & STORE ---
        # Ensure torques_ctrl is a numpy array (can be a list or other sequence coming from ROS)
        self.torques_sim = np.zeros(self.num_sim_dofs)
        try:
            if torques_ctrl is None:
                # Defensive: shouldn't happen because standing_controller returns torques, but handle anyway
                torques_arr = np.zeros(self.num_ctrl_joints, dtype=np.float32)
            else:
                torques_arr = np.asarray(torques_ctrl, dtype=np.float32)

            # Validate shapes before masked assignment
            if torques_arr.ndim == 1 and torques_arr.shape[0] >= np.count_nonzero(self.valid_ctrl_mask):
                # Use boolean mask on controller-ordered torques to get only valid controller joints
                torques_for_valid_ctrl = torques_arr[self.valid_ctrl_mask]
                # Assign to simulation-order indices
                self.torques_sim[self.valid_sim_indices] = torques_for_valid_ctrl
            else:
                # Shape mismatch: fallback or try to broadcast if possible
                print(f"[DogRunner] WARNING: Unexpected torques shape {getattr(torques_arr, 'shape', None)}; expected (>= {np.count_nonzero(self.valid_ctrl_mask)},). Using zeros fallback.")
                # leave self.torques_sim as zeros
        except Exception as e:
            print(f"[DogRunner] ERROR mapping torques to sim indices: {e}")
            # Ensure no exception prevents simulation step; keep zero torques
            self.torques_sim = np.zeros(self.num_sim_dofs)
        
        # (Debug log)
        self.debug_counter += 1
        #if self.debug_counter % 100 == 0:
            #print(f"--- {self.state.name} STATE (Step {self.debug_counter}) ---")
            #print(f"  Sim Loop Duration (ms): {self.sim_dt * 1000.0:.2f} (Fixed)")
            #print(f"  Torques (Sim) Norm: {np.linalg.norm(self.torques_sim)}") 
            #print("-------------------------")

    def step_physics(self):
        """Apply torques and step the simulation."""
        #if self.physics_step_counter % 2 == 0:
        if hasattr(self, 'torques_sim'):
            torques_tensor = torch.tensor(self.torques_sim, dtype=torch.float32, device="cpu").unsqueeze(0)
            self.dog.articulation.set_joint_efforts(torques_tensor)

        self.physics_step_counter += 1  
        print(f"Physics Step Counter: {self.physics_step_counter}")

    def post_physics(self):
        pass

    def reset_condition(self):
        pass

    def shutdown(self):
        """Called from main.py to clean up."""
        print("[DogRunner] Shutting down...")
        self.mpc_controller.shutdown()
        pygame.quit()

                #print("Joint map created:")
        #print(f"  Controller Names: {self.controller_joint_names}")
        #print(f"  Sim Indices:      {self.ctrl_to_sim_idx_np.tolist()}")
        #print("\n[DogRunner] Getting static link masses using DynamicControl...")
        
        # Initialize variables *before* try block
        # self.all_link_mass_map = {}
        # self.joint_to_link_mass_map = {}
        # self.leg_link_masses = {} 
        # self.static_mass_diag_ctrl_order = np.zeros(self.num_ctrl_joints)
        
        # try:
        #     # Get all body names *directly from the articulation*
        #     # We already have self.art_handle from earlier in __init__
        #     if self.art_handle == _dynamic_control.INVALID_HANDLE:
        #         print("[DogRunner] ERROR: Invalid articulation handle. Cannot get masses.")
        #     else:
        #         # This list is populated by self.dog.articulation.initialize()
        #         all_body_paths = self.dog.articulation.body_names 
        #         if not all_body_paths:
        #             print("[DogRunner] ERROR: Articulation has no body names.")
        #         else:
        #             print(f"  Found {len(all_body_paths)} rigid bodies in articulation.")
                    
        #             # --- FIX: Iterate using the index (i) ---
        #             for i, body_path in enumerate(all_body_paths):
        #                 # body_path is the full path, e.g., "/World/Go2/FL_hip"
        #                 # link_name is the short name, e.g., "FL_hip"
        #                 link_name = body_path.split('/')[-1] 
                        
        #                 # --- FIX: Use the correct function get_articulation_body(handle, index) ---
        #                 body_handle = self._dc.get_articulation_body(self.art_handle, i)
                        
        #                 if body_handle != _dynamic_control.INVALID_HANDLE:
        #                     mass_props = self._dc.get_rigid_body_properties(body_handle)
        #                     if mass_props is not None:
        #                         self.all_link_mass_map[link_name] = mass_props.mass
        #                     else:
        #                         print(f"    - WARNING: Could not get mass properties for body '{link_name}'")
        #                         self.all_link_mass_map[link_name] = 0.0
        #                 else:
        #                     print(f"    - WARNING: Could not get body handle for '{link_name}' (index {i})")
        #                     self.all_link_mass_map[link_name] = 0.0

        #             # --- Now, populate the joint-to-mass map ---
        #             print("  Mapping controller joints to their corresponding link masses:")
        #             for joint_name in self.controller_joint_names:
        #                 link_name = joint_name.replace('_joint', '')
                        
        #                 if link_name in self.all_link_mass_map:
        #                     mass = self.all_link_mass_map[link_name]
        #                     self.joint_to_link_mass_map[joint_name] = mass
        #                     print(f"    - {joint_name} (moves link: {link_name}): {mass:.4f} kg")
        #                 else:
        #                     self.joint_to_link_mass_map[joint_name] = 0.0 # Store 0.0 if not found
        #                     print(f"    - WARNING: Could not find mass for link '{link_name}' (from joint '{joint_name}')")

        #             # --- Create the 12-element static mass vector ---
        #             mass_vector = []
        #             for joint_name in self.controller_joint_names:
        #                 mass_vector.append(self.joint_to_link_mass_map.get(joint_name, 0.0))
        #             self.static_mass_diag_ctrl_order = np.array(mass_vector)
                    
        #             print(f"\n  Static mass vector (controller order): {self.static_mass_diag_ctrl_order.tolist()}\n")
                    
        #             # (Logging)
        #             self.leg_link_masses = {
        #                 'FL': [self.joint_to_link_mass_map.get('FL_hip_joint', 0),
        #                        self.joint_to_link_mass_map.get('FL_thigh_joint', 0),
        #                        self.joint_to_link_mass_map.get('FL_calf_joint', 0)],
        #                 'FR': [self.joint_to_link_mass_map.get('FR_hip_joint', 0),
        #                        self.joint_to_link_mass_map.get('FR_thigh_joint', 0),
        #                        self.joint_to_link_mass_map.get('FR_calf_joint', 0)],
        #                 'RL': [self.joint_to_link_mass_map.get('RL_hip_joint', 0),
        #                        self.joint_to_link_mass_map.get('RL_thigh_joint', 0),
        #                        self.joint_to_link_mass_map.get('RL_calf_joint', 0)],
        #                 'RR': [self.joint_to_link_mass_map.get('RR_hip_joint', 0),
        #                        self.joint_to_link_mass_map.get('RR_thigh_joint', 0),
        #                        self.joint_to_link_mass_map.get('RR_calf_joint', 0)]
        #             }
        #             print("\n  Static masses per leg (Hip, Thigh, Calf):")
        #             print(f"  FL: {self.leg_link_masses['FL']}")
        #             print(f"  FR: {self.leg_link_masses['FR']}")
        #             print(f"  RL: {self.leg_link_masses['RL']}")
        #             print(f"  RR: {self.leg_link_masses['RR']}\n")

        # except Exception as e:
        #     print(f"[DogRunner] ERROR during link mass retrieval using DynamicControl: {e}")

                # try:
        #     self.mpc_controller.publish_state(state_data)
        # except Exception as e: print(f"[DogRunner] Error calling mpc_controller.publish_state: {e}")

        # if self.state == SimulationState.INIT:
        #     torques_ctrl = self.standing_controller.get_torques(current_pos_ctrl, current_vel_ctrl)
        # elif self.state == SimulationState.ACTIVE:
        #     try:
        #         self.mpc_controller.publish_state(state_data)
        #         torques_ctrl = self.mpc_controller.get_action(target_timestamp=current_sim_time)
        #         print(f"[DogRunner] Received torques from MPC controller: {torques_ctrl}")
        #         if torques_ctrl is None:
        #             print("[DogRunner] No new torque command received from ROS MPC controller.")
        #             raise ValueError("No new torque command received")
        #     except Exception as e:
        #         print(f"[DogRunner] Error calling mpc_controller.get_action: {e}")
        # else:
        #     print(f"[DogRunner] WARNING: Unknown simulation state '{self.state}'. Using standing controller as fallback.")

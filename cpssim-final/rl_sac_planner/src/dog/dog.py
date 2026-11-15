from pxr import Gf, Sdf, UsdPhysics, PhysxSchema
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage 
from isaacsim.core.utils.prims import get_prim_at_path

try:
    from isaacsim.sensors.physics import ContactSensor
    from isaacsim.sensors.camera import Camera
except Exception:
    # Backward compatibility for older Isaac Sim releases.
    from omni.isaac.sensor import ContactSensor, Camera


import torch
import numpy as np
import os
import sys

default_asset_path = "/home/mena/cpssim2/IsaacLabFromTemu/assets/robot/go2/go2.usd"

class Dog:
    def __init__(self, stage, asset_path=default_asset_path):
        self.asset_path = asset_path
        self.stage = stage
        self.articulation = None
        self.foot_sensors = {}
        self._contact_sensors_initialized = False
        self.camera = None

        if not os.path.exists(self.asset_path):
            print(f"--- FATAL ERROR: Asset file not found at {self.asset_path}")
            sys.exit()

    def add_Dog(self, world, name="Go2"):
        self.name = name
        self.prim_root = f"/World/{self.name}"
        self.articulation_root = f"{self.prim_root}/base"

        print(f"[Dog] Adding reference to {self.asset_path}")
        add_reference_to_stage(usd_path=self.asset_path, prim_path=self.prim_root)
        
        self.articulation = Articulation(prim_paths_expr=self.articulation_root)
        world.scene.add(self.articulation)

        # --- SENSOR & PHYSICS SETUP ---
        print("[Dog] Configuring Contact Reporting...")
        
        for leg_name in ['FL', 'FR', 'RL', 'RR']:
            foot_path = f"{self.prim_root}/{leg_name}_foot"
            sensor_path = f"{foot_path}/ContactSensor"
            
            # 1. APPLY CONTACT REPORT API
            # This tells PhysX to actually calculate and store force data for this link
            foot_prim = get_prim_at_path(foot_path)
            if not foot_prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(foot_prim)
            
            # 2. CREATE THE SENSOR
            # radius=0.03 covers the Go2 foot pad effectively
            sensor = ContactSensor(
                prim_path=sensor_path,
                name=f"{leg_name}_contact",
                min_threshold=0.1,
                max_threshold=1000000.0,
                radius=0.03,
                translation=np.array([0, 0, 0])
            )
            self.foot_sensors[leg_name] = sensor

        # NOTE:
        # Do NOT call world.reset() here. The runner resets the world after the full scene is assembled.
        # If we reset here and then reset again later, the sensor handles get invalidated and will report zeros.
        self._contact_sensors_initialized = False
        print("[Dog] Contact sensors created (pending initialization after world reset).")

        camera_path = f"{self.articulation_root}/camera"
        self.camera = Camera(
            prim_path=camera_path,
            name="front_camera",
            resolution=(640, 480),      
            frequency=30,
            orientation=[1,0,0,0]
        )


    def initialize_sensors(self):
        """Initialize contact sensors after the final world reset/first step."""
        if self._contact_sensors_initialized:
            return
        for sensor in self.foot_sensors.values():
            sensor.initialize()
            sensor.add_raw_contact_data_to_frame()
            sensor.resume()
        self._contact_sensors_initialized = True
        print("[Dog] Contact sensors initialized and resumed.")
        #self.camera.initialize()



    def actuate(self, torques_np: np.ndarray):
        if self.articulation:
            torques = torch.tensor(torques_np, dtype=torch.float32, device="cpu").unsqueeze(0)
            self.articulation.set_joint_efforts(torques)
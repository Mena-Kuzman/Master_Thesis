from dataclasses import dataclass
import numpy as np

from src.base.baseWorld import BaseWorld

from src.dog.dog import Dog
from isaacsim.core.api.objects import FixedCuboid
from isaacsim.core.api.materials import PhysicsMaterial
from pxr import UsdPhysics, PhysxSchema


class DogWorld(BaseWorld):

    @dataclass
    class Params(BaseWorld.Params):
        add_dog : bool = True
        add_camera : bool = True
        add_obstacles : bool = False
        add_stairs : bool = True
        stair_start_x : float = 2.5      # Starting position (meters)
        stair_step_height : float = 0.10 # Height of each step (meters)
        stair_step_width : float = 0.30  # Width of each step (meters)
        stair_step_depth : float = 5.0   # Depth (Y-axis) of each step (meters)
        stair_num_steps_up : int = 3     # Number of steps going up
        stair_num_steps_down : int = 3   # Number of steps going down


    def __init__(self, params : Params):
        super().__init__(params)
        self.params = params

        if self.params.add_dog == True:
            #init the dog
            self.dog = Dog(self.stage)
            self.dog.add_Dog(self.world)
        
        if self.params.add_stairs == True:
            self._add_stairs()
    
    def _add_stairs(self):
        """Add upstairs and downstairs to the simulation environment."""
        step_height = self.params.stair_step_height
        step_width = self.params.stair_step_width
        step_depth = self.params.stair_step_depth
        start_x = self.params.stair_start_x
        num_steps_up = self.params.stair_num_steps_up
        num_steps_down = self.params.stair_num_steps_down
        
        # Create material for stairs (good friction)
        stair_material_path = "/World/Physics_Materials/StairMaterial"
        stair_material = PhysicsMaterial(
            stair_material_path,
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.0
        )
        stair_material_prim = self.stage.GetPrimAtPath(stair_material_path)
        PhysxSchema.PhysxMaterialAPI.Apply(stair_material_prim).CreateFrictionCombineModeAttr().Set("max")
        
        # Create UPSTAIRS
        for i in range(num_steps_up):
            # Position each step
            x_pos = float(start_x + i * step_width)
            z_pos = float(step_height / 2 + i * step_height)  # Stack height
            
            # Step dimensions: width x depth x height
            step = FixedCuboid(
                prim_path=f"/World/stairs_up_step_{i}",
                position=np.array([x_pos, 0.0, z_pos], dtype=np.float64),
                size=1.0,
                scale=np.array([float(step_width), float(step_depth), float(step_height)], dtype=np.float64),
                color=np.array([0.0, 0.0, 0.0], dtype=np.float32)
            )
            step.apply_physics_material(stair_material)
        
        # Create DOWNSTAIRS (after the upstairs)
        down_start_x = float(start_x + num_steps_up * step_width)
        initial_height = float(step_height * num_steps_up)
        
        for i in range(num_steps_down):
            # Position each step going down
            x_pos = float(down_start_x + i * step_width)
            z_pos = float(initial_height - (step_height / 2 + i * step_height))
            
            # Step dimensions
            step = FixedCuboid(
                prim_path=f"/World/stairs_down_step_{i}",
                position=np.array([x_pos, 0.0, z_pos], dtype=np.float64),
                size=1.0,
                scale=np.array([float(step_width), float(step_depth), float(step_height)], dtype=np.float64),
                color=np.array([0.0, 0.0, 0.0], dtype=np.float32)
            )
            step.apply_physics_material(stair_material)
        
        print(f"[DogWorld] Added stairs: {num_steps_up} steps up, {num_steps_down} steps down (depth={step_depth}m)")

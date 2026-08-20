from dataclasses import dataclass

from isaacsim.core.api import World, SimulationContext
from isaacsim.core.utils.prims import create_prim

import omni.usd

from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.core.api.objects import GroundPlane
from pxr import UsdPhysics, PhysxSchema, Gf

class BaseWorld:

    @dataclass
    class Params:
        add_light: bool = True
        add_ground_plane : bool = True
        ground_plane_z_position : float = -0.0
        ground_static_friction: float = 0.5  # Match MuJoCo mu=0.5
        ground_dynamic_friction: float = 0.5  # Match MuJoCo mu=0.5
        ground_restitution: float = 0.0  # No bounce
        
        

    def __init__(self, params : Params):
        self.params = params

        self.stage = omni.usd.get_context().get_stage()
        self.world = World(stage_units_in_meters=1.0, physics_dt=0.002, rendering_dt=0.002)  # physics_dt set to 0.002 for 500 Hz

        self.context = SimulationContext()


        self.previous_time = self.context.current_time

        if self.params.add_ground_plane == True:
            self.world.scene.add_default_ground_plane(z_position=self.params.ground_plane_z_position)
            material_path = "/World/Physics_Materials/GroundMaterial"
            ground_material = PhysicsMaterial(
                material_path,
                static_friction=self.params.ground_static_friction,
                dynamic_friction=self.params.ground_dynamic_friction,
                restitution=self.params.ground_restitution
            )
            material_prim = omni.usd.get_context().get_stage().GetPrimAtPath(material_path)
            PhysxSchema.PhysxMaterialAPI.Apply(material_prim).CreateFrictionCombineModeAttr().Set("max")
            ground_object = GroundPlane(prim_path="/World/defaultGroundPlane")
            ground_object.apply_physics_material(ground_material)


        # Add camera-attached light if requested
        if self.params.add_light:
            # Remove default light if it exists
            if self.stage.GetPrimAtPath("/World/defaultLight"):
                self.stage.RemovePrim("/World/defaultLight")
            # Add a new directional light at camera pose
            cam_prim = self.stage.GetPrimAtPath("/World/camera")
            if cam_prim:
                cam_xform = cam_prim.GetAttribute("xformOp:transform")
                if cam_xform:
                    cam_mat = cam_xform.Get()
                    create_prim("/World/cameraLight", "DistantLight", attributes={
                        "inputs:intensity": 2000.0,
                        "inputs:color": (1.0, 1.0, 1.0)
                    })
            else:
                # Fallback: add a default distant light
                create_prim("/World/cameraLight", "DistantLight", attributes={
                    "inputs:intensity": 2000.0,
                    "inputs:color": (1.0, 1.0, 1.0)
                })

        # Configure PhysX scene (solver and contact parameters) if it already exists.
        # The Isaac `World` helper typically creates `/World/physicsScene` for us;
        # to avoid conflicting configuration, only tune an existing scene.
        # physics_prim_path = "/World/physicsScene"
        # physics_prim = self.stage.GetPrimAtPath(physics_prim_path)
        # if physics_prim and physics_prim.IsValid():
        #     # Wrap existing prim with USD physics scene schema (no new prims created)
        #     scene_schema = UsdPhysics.Scene(physics_prim)
        #     if scene_schema:
        #         # Set gravity explicitly (keeps defaults consistent and readable in USD).
        #         scene_schema.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        #         scene_schema.CreateGravityMagnitudeAttr().Set(9.81)

        #     # Apply PhysX-specific tuning on the same prim
        #     physx_scene = PhysxSchema.PhysxSceneAPI.Apply(physics_prim)

        #     # Match World physics_dt (0.002s -> 500 Hz) and increase solver iterations for stability.
        #     # Attribute names depend on PhysX USD schema version; use Max* variants here.
        #     physx_scene.CreateTimeStepsPerSecondAttr().Set(500.0)
        #     physx_scene.CreateMaxPositionIterationCountAttr().Set(8)
        #     physx_scene.CreateMaxVelocityIterationCountAttr().Set(2)


    #def get_dt(self):
    #    _dt = self.context.current_time - self.previous_time 
    #    self.previous_time = self.context.current_time
    #    return _dt

    def initialise(self):
        """User defined initialisation function called once after world creation. CAll it yourself if you need """
        pass

























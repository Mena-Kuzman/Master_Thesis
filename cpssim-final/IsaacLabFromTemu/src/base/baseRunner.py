from src.base.baseWorld import BaseWorld
import time


class BaseRunner():
    def __init__(self, world : BaseWorld):
        self.stage = world
        self.stage.world.reset()

    def pre_physics(self):
        #get states, controller
        pass

    def step_physics(self):
        #apply forces, torques, target vel or pos, etc
        pass

    def post_physics(self):
        #capture img, prnt data, etc.
        pass

    def reset_condition(self):
        #returns a booleean
        pass

    def reset(self):
        self.stage.world.reset()


    def run(self):
        self.render_every_n_steps = 8 # 30 fps 
        self.step_count = 0
        while True:
            # 1. Get state, decide action
            # (This calls DogRunner's pre_physics)
            self.pre_physics() 
            
            # 2. Apply action
            # (This calls DogRunner's "slim" step_physics)
            self.step_physics() 
            
            # 3. Step the simulation
            self.step_count += 1
            
            if self.step_count % self.render_every_n_steps == 0:
                # This is a render frame
                self.stage.world.step(render=True)
                self.step_count = 0 # Reset counter
            else:
                # This is a physics-only frame
                self.stage.world.step(render=False)

            # 4. Do post-step logic
            # (This calls DogRunner's post_physics)
            self.post_physics()



        #    if(self.reset_condition() == True):
        #        self.reset()



from dataclasses import dataclass
import random

class RandomController:
    @dataclass
    class Params:
        pass
    def __init__(self, params: Params):
        self.domain = [-1.0, 1.0]
        self.n_dim = 12
    def get_action(self):
        return [random.uniform(self.domain[0], self.domain[1]) for _ in range(self.n_dim)]
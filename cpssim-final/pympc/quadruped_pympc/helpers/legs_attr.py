from __future__ import annotations

import numpy as np


class LegsAttr:
    """Container for per-leg values in FL/FR/RL/RR order."""

    def __init__(self, FL, FR, RL, RR):
        self.FL = FL
        self.FR = FR
        self.RL = RL
        self.RR = RR
        self.order = ["FL", "FR", "RL", "RR"]

    def to_list(self, order=None):
        order = order if order is not None else self.order
        return [getattr(self, leg) for leg in order]

    def __getitem__(self, key):
        if key not in self.order:
            raise KeyError(f"Key {key} is not a valid leg label. Expected any of {self.order}")
        return getattr(self, key)

    def __setitem__(self, key, value):
        if key not in self.order:
            raise KeyError(f"Key {key} is not a valid leg label. Expected any of {self.order}")
        setattr(self, key, value)

    def __iter__(self):
        return iter(self.to_list())

    def __add__(self, other):
        if isinstance(other, LegsAttr):
            return LegsAttr(
                FL=self.FL + other.FL,
                FR=self.FR + other.FR,
                RL=self.RL + other.RL,
                RR=self.RR + other.RR,
            )
        if isinstance(other, (int, float, np.ndarray)):
            return LegsAttr(
                FL=self.FL + other,
                FR=self.FR + other,
                RL=self.RL + other,
                RR=self.RR + other,
            )
        return NotImplemented

    def __str__(self):
        parts = []
        for leg in self.order:
            val = getattr(self, leg)
            if isinstance(val, np.ndarray):
                formatted_val = np.array2string(val, precision=3, suppress_small=True, separator=", ")
            else:
                formatted_val = str(val)
            parts.append(f"{leg}={formatted_val}")
        return "; ".join(parts)

    def __repr__(self):
        return self.__str__()

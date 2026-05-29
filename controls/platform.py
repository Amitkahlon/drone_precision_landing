import numpy as np


class Platform:
    def __init__(
            self,
            width: float = 1.0,
            depth: float = 1.0,
            thickness: float = 0.05,
            color: tuple[float, float, float, float] = (0.8, 0.2, 0.2, 1.0),
    ):
        self.width = width
        self.depth = depth
        self.thickness = thickness
        self.color = color


class Waypoint:
    def __init__(self, position: tuple[float, float, float], speed: float):
        self.position = np.array(position, dtype=float)
        self.speed = float(speed)

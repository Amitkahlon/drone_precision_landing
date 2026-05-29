import numpy as np

from .platform import Platform, Waypoint
from .moving_platform import MovingPlatform


class Mission:
    def __init__(self, start: tuple, platform: Platform):
        self.start     = np.array(start, dtype=float)
        self._platform = platform
        self._checkpoints: list[Waypoint] = []

    def add_checkpoint(self, position: tuple, speed: float) -> 'Mission':
        self._checkpoints.append(Waypoint(position, speed))
        return self

    def build_platform(self) -> MovingPlatform:
        if len(self._checkpoints) < 2:
            raise ValueError("Mission needs at least 2 checkpoints to build a moving platform")
        return MovingPlatform(self._platform, self._checkpoints)

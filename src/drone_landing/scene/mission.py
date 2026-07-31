import numpy as np

from .platform import MovingPlatform, Platform, Waypoint


class Mission:
    """A scenario definition: where the drone starts and the route its target flies.

    Holds no simulation state, so the same mission can be flown repeatedly.
    """

    def __init__(self, start: tuple, platform: Platform):
        self.start = np.array(start, dtype=float)
        self._platform = platform
        self._checkpoints: list[Waypoint] = []

    def add_checkpoint(self, position: tuple, speed: float) -> 'Mission':
        """Append a waypoint, where `speed` is flown on the leg leaving it."""
        self._checkpoints.append(Waypoint(position, speed))
        return self

    def build_moving_platform(self) -> MovingPlatform:
        if len(self._checkpoints) < 2:
            raise ValueError("Mission needs at least 2 checkpoints to build a moving platform")
        return MovingPlatform(self._platform, self._checkpoints)

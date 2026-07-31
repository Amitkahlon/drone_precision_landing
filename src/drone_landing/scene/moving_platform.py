import mujoco
import numpy as np

from .platform import Platform, Waypoint


class MovingPlatform:
    def __init__(self, platform: Platform, waypoints: list[Waypoint]):
        if len(waypoints) < 2:
            raise ValueError("MovingPlatform requires at least 2 waypoints")
        self.platform = platform
        self.waypoints = waypoints
        self._idx = 0
        self._pos = waypoints[0].position.copy()
        self._mocap_id: int | None = None
        self._data: mujoco.MjData | None = None

    @property
    def position(self) -> np.ndarray:
        return self._pos.copy()

    @property
    def velocity(self) -> np.ndarray:
        next_idx = (self._idx + 1) % len(self.waypoints)
        target = self.waypoints[next_idx].position
        direction = target - self._pos
        distance = np.linalg.norm(direction)
        if distance < 1e-6:
            return np.zeros(3)
        return (direction / distance) * self.waypoints[self._idx].speed

    def _bind(self, mocap_id: int, data: mujoco.MjData) -> None:
        self._mocap_id = mocap_id
        self._data = data
        self._data.mocap_pos[self._mocap_id] = self._pos

    def step(self, dt: float) -> None:
        next_idx = (self._idx + 1) % len(self.waypoints)
        target = self.waypoints[next_idx].position
        direction = target - self._pos
        distance = np.linalg.norm(direction)
        move = self.waypoints[self._idx].speed * dt

        if distance <= move:
            self._pos = target.copy()
            self._idx = next_idx
        elif distance > 0:
            self._pos += (direction / distance) * move

        if self._mocap_id is not None:
            self._data.mocap_pos[self._mocap_id] = self._pos

import mujoco
import numpy as np


class Platform:
    """Physical dimensions and colour of a landing platform."""

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
    """A point on a platform's route, carrying the speed of the leg leaving it."""

    def __init__(self, position: tuple[float, float, float], speed: float):
        self.position = np.array(position, dtype=float)
        self.speed = float(speed)


class MovingPlatform:
    """A platform that cycles through its waypoints forever.

    Driven as a MuJoCo mocap body, so it moves on rails and is unaffected by the
    drone touching down on it.
    """

    def __init__(self, platform: Platform, waypoints: list[Waypoint]):
        if len(waypoints) < 2:
            raise ValueError("MovingPlatform requires at least 2 waypoints")
        self.platform = platform
        self.waypoints = waypoints
        self._index = 0
        self._position = waypoints[0].position.copy()
        self._mocap_id: int | None = None
        self._data: mujoco.MjData | None = None

    @property
    def position(self) -> np.ndarray:
        return self._position.copy()

    @property
    def velocity(self) -> np.ndarray:
        """Current heading scaled by the speed of the leg being flown."""
        direction, distance = self._to_next_waypoint()
        if distance < 1e-6:
            return np.zeros(3)
        return (direction / distance) * self.waypoints[self._index].speed

    def bind(self, mocap_id: int, data: mujoco.MjData) -> None:
        """Attach to the mocap body the scene builder created for this platform."""
        self._mocap_id = mocap_id
        self._data = data
        self._data.mocap_pos[self._mocap_id] = self._position

    def step(self, dt: float) -> None:
        direction, distance = self._to_next_waypoint()
        travel = self.waypoints[self._index].speed * dt
        next_index = self._next_index()

        if distance <= travel:
            # Snap to the waypoint rather than overshooting, and start the next leg.
            self._position = self.waypoints[next_index].position.copy()
            self._index = next_index
        elif distance > 0:
            self._position += (direction / distance) * travel

        if self._data is not None:
            self._data.mocap_pos[self._mocap_id] = self._position

    def _next_index(self) -> int:
        return (self._index + 1) % len(self.waypoints)

    def _to_next_waypoint(self) -> tuple[np.ndarray, float]:
        direction = self.waypoints[self._next_index()].position - self._position
        return direction, float(np.linalg.norm(direction))

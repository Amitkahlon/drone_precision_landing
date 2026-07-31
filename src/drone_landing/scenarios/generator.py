import random
from dataclasses import dataclass, field

import numpy as np

from ..scene import Mission, Platform, Waypoint
from ..settings import WAYPOINT_SPACING_ATTEMPTS
from .config import ScenarioConfig


@dataclass
class Scenario:
    """A generated mission definition: platform waypoints plus per-leg speeds."""

    seed: int
    waypoints: list[Waypoint]
    start: tuple[float, float, float] = (0.0, 0.0, 0.0)
    platform: Platform = field(default_factory=lambda: Platform(color=(0.9, 0.6, 0.1, 1.0)))

    def build_mission(self) -> Mission:
        mission = Mission(start=self.start, platform=self.platform)
        for waypoint in self.waypoints:
            mission.add_checkpoint(tuple(waypoint.position), speed=waypoint.speed)
        return mission

    @property
    def legs(self) -> list[dict]:
        """One entry per leg of the cyclic path, each carrying the speed it is flown at."""
        legs = []
        for index, waypoint in enumerate(self.waypoints):
            next_index = (index + 1) % len(self.waypoints)
            next_waypoint = self.waypoints[next_index]
            legs.append({
                "from": index,
                "to": next_index,
                "speed": waypoint.speed,
                "length": float(np.linalg.norm(next_waypoint.position - waypoint.position)),
            })
        return legs

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "start": list(self.start),
            "num_waypoints": len(self.waypoints),
            "waypoints": [
                {"position": wp.position.tolist(), "speed": wp.speed} for wp in self.waypoints
            ],
            "legs": self.legs,
        }


def generate_scenario(config: ScenarioConfig, seed: int) -> Scenario:
    """Sample a platform route reproducibly from `seed`."""
    rng = random.Random(seed)
    count = rng.randint(config.min_waypoints, config.max_waypoints)

    positions: list[np.ndarray] = []
    for _ in range(count):
        positions.append(_sample_position(rng, config, positions))

    waypoints = [
        Waypoint(tuple(position), speed=rng.uniform(config.min_speed, config.max_speed))
        for position in positions
    ]
    return Scenario(seed=seed, waypoints=waypoints, start=config.start)


def _sample_position(
        rng: random.Random,
        config: ScenarioConfig,
        placed: list[np.ndarray],
) -> np.ndarray:
    """Sample a waypoint at least min_spacing (in XY) from the ones already placed.

    Falls back to the best-separated candidate after a fixed number of attempts, so a
    bounding box that is too tight for the requested spacing degrades instead of failing.
    """
    best: np.ndarray | None = None
    best_clearance = -1.0

    for _ in range(WAYPOINT_SPACING_ATTEMPTS):
        candidate = np.array([
            rng.uniform(-config.bounds_x, config.bounds_x),
            rng.uniform(-config.bounds_y, config.bounds_y),
            rng.uniform(config.min_altitude, config.max_altitude),
        ])
        clearance = _clearance(candidate, placed)
        if clearance >= config.min_spacing:
            return candidate
        if clearance > best_clearance:
            best, best_clearance = candidate, clearance

    return best


def _clearance(candidate: np.ndarray, placed: list[np.ndarray]) -> float:
    """XY distance to the nearest already-placed waypoint."""
    if not placed:
        return float("inf")
    return min(float(np.linalg.norm(candidate[:2] - position[:2])) for position in placed)

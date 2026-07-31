import random
from dataclasses import dataclass, field

import numpy as np

from ..scene import Mission, Platform, Waypoint
from ..settings import WAYPOINT_SPACING_ATTEMPTS


@dataclass
class ScenarioConfig:
    """Bounds and ranges the generator samples a platform trajectory from."""

    bounds_x: float = 8.0
    bounds_y: float = 8.0
    min_waypoints: int = 3
    max_waypoints: int = 8
    min_speed: float = 0.5
    max_speed: float = 2.0
    min_altitude: float = 0.5
    max_altitude: float = 0.5
    min_spacing: float = 2.0
    start: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def to_dict(self) -> dict:
        return {
            "bounds_x": self.bounds_x,
            "bounds_y": self.bounds_y,
            "min_waypoints": self.min_waypoints,
            "max_waypoints": self.max_waypoints,
            "min_speed": self.min_speed,
            "max_speed": self.max_speed,
            "min_altitude": self.min_altitude,
            "max_altitude": self.max_altitude,
            "min_spacing": self.min_spacing,
            "start": list(self.start),
        }


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
        for i, waypoint in enumerate(self.waypoints):
            nxt = self.waypoints[(i + 1) % len(self.waypoints)]
            legs.append({
                "from": i,
                "to": (i + 1) % len(self.waypoints),
                "speed": waypoint.speed,
                "length": float(np.linalg.norm(nxt.position - waypoint.position)),
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
    rng = random.Random(seed)
    count = rng.randint(config.min_waypoints, config.max_waypoints)

    positions: list[np.ndarray] = []
    for _ in range(count):
        positions.append(_sample_position(rng, config, positions))

    waypoints = [
        Waypoint(tuple(pos), speed=rng.uniform(config.min_speed, config.max_speed))
        for pos in positions
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
    if not placed:
        return float("inf")
    return min(float(np.linalg.norm(candidate[:2] - p[:2])) for p in placed)

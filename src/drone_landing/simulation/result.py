from dataclasses import dataclass

import numpy as np

from ..drone import Drone
from ..enums import DroneState
from ..scene import MovingPlatform


@dataclass
class RunResult:
    """Outcome of a single mission.

    `landed` and `on_platform` are kept apart because the controller's own
    touchdown check is Z-only: a drone that drifts off the platform during
    descent still reports GROUNDED, so only their conjunction is a success.
    """

    landed: bool  # the controller's own criterion: it reached DroneState.GROUNDED
    on_platform: bool  # touchdown happened inside the platform footprint
    success: bool
    failure_reason: str | None
    final_state: str
    duration_s: float
    steps: int
    xy_error_m: float | None
    drone_position: list[float | None]
    platform_position: list[float | None]

    def to_dict(self) -> dict:
        return {
            "landed": self.landed,
            "on_platform": self.on_platform,
            "success": self.success,
            "failure_reason": self.failure_reason,
            "final_state": self.final_state,
            "duration_s": self.duration_s,
            "steps": self.steps,
            "xy_error_m": self.xy_error_m,
            "drone_position": self.drone_position,
            "platform_position": self.platform_position,
        }


def build_result(
        drone: Drone,
        target: MovingPlatform,
        state: DroneState,
        steps: int,
        dt: float,
        aborted: bool,
        diverged: bool,
) -> RunResult:
    drone_pos = np.array([drone.get_x(), drone.get_y(), drone.get_z()])
    platform_pos = target.position

    delta = drone_pos[:2] - platform_pos[:2]
    on_platform = (
            abs(delta[0]) <= target.platform.width / 2
            and abs(delta[1]) <= target.platform.depth / 2
    )
    landed = state == DroneState.GROUNDED
    success = landed and on_platform

    return RunResult(
        landed=landed,
        on_platform=bool(on_platform),
        success=bool(success),
        failure_reason=_failure_reason(success, landed, aborted, diverged),
        final_state=state.name,
        duration_s=round(steps * dt, 3),
        steps=steps,
        xy_error_m=finite_or_none(np.linalg.norm(delta)),
        drone_position=[finite_or_none(value) for value in drone_pos],
        platform_position=[finite_or_none(value) for value in platform_pos],
    )


def _failure_reason(
        success: bool,
        landed: bool,
        aborted: bool,
        diverged: bool,
) -> str | None:
    if success:
        return None
    if landed:
        return "off_platform"
    if aborted:
        return "aborted"
    if diverged:
        return "diverged"
    return "timeout"


def finite_or_none(value) -> float | None:
    """None rather than NaN/inf, so a diverged run still serialises to valid JSON."""
    value = float(value)
    return round(value, 4) if np.isfinite(value) else None

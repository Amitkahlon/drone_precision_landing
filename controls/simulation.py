import os
import time
from dataclasses import dataclass

import mujoco
import mujoco.viewer
import numpy as np

from .drone import Drone
from .drone_controller import DroneController
from .enums import DroneState
from .mission import Mission
from .moving_platform import MovingPlatform
from .scene_builder import SceneBuilder

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "models", "drone.xml"
)

_WORLD_LIMIT = 25.0  # floor plane is 20x20, so beyond this the drone is unrecoverable


@dataclass
class RunResult:
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


def run_mission(
        mission: Mission,
        model_path: str = DEFAULT_MODEL_PATH,
        *,
        hover_altitude: float = 3.0,
        timeout: float = 120.0,
        viewer: bool = False,
) -> RunResult:
    """Fly a mission end to end and report whether the drone landed on the platform.

    Headless by default so batches run far faster than real time; viewer=True replays a
    single scenario in the passive viewer at wall-clock speed.
    """
    target = mission.build_platform()

    scene = SceneBuilder(model_path)
    scene.add_moving_platform("target", target)
    model, data = scene.build()

    drone = Drone(model, data)
    controller = DroneController(drone)

    scene.add_drone("main", drone)
    scene.apply_mission(mission)

    controller.set_target(target)
    controller.take_off(hover_altitude)

    dt = model.opt.timestep
    max_steps = int(timeout / dt)
    steps = 0
    aborted = False

    def advance() -> None:
        controller.step()
        scene.step(dt)
        mujoco.mj_step(model, data)

    def flying() -> bool:
        return steps < max_steps and controller.state != DroneState.GROUNDED and not _diverged(drone)

    if viewer:
        with mujoco.viewer.launch_passive(model, data) as v:
            while flying():
                if not v.is_running():
                    aborted = True
                    break
                step_start = time.time()

                advance()
                steps += 1

                _update_label(v, drone, controller.state.name)
                v.sync()

                remaining = dt - (time.time() - step_start)
                if remaining > 0:
                    time.sleep(remaining)
    else:
        while flying():
            advance()
            steps += 1

    return _build_result(drone, target, controller.state, steps, dt, aborted)


def _build_result(
        drone: Drone,
        target: MovingPlatform,
        state: DroneState,
        steps: int,
        dt: float,
        aborted: bool,
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

    if success:
        failure_reason = None
    elif landed:
        failure_reason = "off_platform"
    elif aborted:
        failure_reason = "aborted"
    elif _diverged(drone):
        failure_reason = "diverged"
    else:
        failure_reason = "timeout"

    return RunResult(
        landed=landed,
        on_platform=bool(on_platform),
        success=bool(success),
        failure_reason=failure_reason,
        final_state=state.name,
        duration_s=round(steps * dt, 3),
        steps=steps,
        xy_error_m=_finite(np.linalg.norm(delta)),
        drone_position=[_finite(v) for v in drone_pos],
        platform_position=[_finite(v) for v in platform_pos],
    )


def _finite(value) -> float | None:
    """None rather than NaN/inf, so a diverged run still serialises to valid JSON."""
    value = float(value)
    return round(value, 4) if np.isfinite(value) else None


def _diverged(drone: Drone) -> bool:
    pos = drone.sensors.pos
    return not np.all(np.isfinite(pos)) or bool(np.max(np.abs(pos)) > _WORLD_LIMIT)


def _update_label(viewer, drone: Drone, label: str) -> None:
    viewer.user_scn.ngeom = 1
    g = viewer.user_scn.geoms[0]
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.zeros(3), np.zeros(3), np.eye(3).flatten(),
        np.array([0.0, 0.0, 0.0, 0.0]),
    )
    g.pos[:] = [drone.get_x(), drone.get_y(), drone.get_z() + 0.4]
    g.size[:] = [0.01, 0.01, 0.01]
    g.label = label

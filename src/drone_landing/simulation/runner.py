from pathlib import Path

import mujoco
import numpy as np

from ..control import DroneController
from ..drone import Drone
from ..enums import DroneState
from ..scene import Mission, SceneBuilder
from ..settings import (
    DEFAULT_HOVER_ALTITUDE_M,
    DEFAULT_TIMEOUT_S,
    MODEL_PATH,
    WORLD_LIMIT_M,
)
from .result import RunResult, build_result
from .viewer import draw_state_label, run_viewer_loop

_TARGET_KEY = "target"
_DRONE_KEY = "main"


def run_mission(
        mission: Mission,
        model_path: str | Path = MODEL_PATH,
        *,
        hover_altitude: float = DEFAULT_HOVER_ALTITUDE_M,
        timeout: float = DEFAULT_TIMEOUT_S,
        viewer: bool = False,
) -> RunResult:
    """Fly a mission end to end and report whether the drone landed on the platform.

    Headless by default so batches run far faster than real time; viewer=True replays a
    single scenario in the passive viewer at wall-clock speed.
    """
    target = mission.build_platform()

    scene = SceneBuilder(model_path)
    scene.add_moving_platform(_TARGET_KEY, target)
    model, data = scene.build()

    drone = Drone(model, data)
    controller = DroneController(drone)

    scene.add_drone(_DRONE_KEY, drone)
    scene.apply_mission(mission)

    controller.set_target(target)
    controller.take_off(hover_altitude)

    timestep = model.opt.timestep
    max_steps = int(timeout / timestep)
    steps = 0

    def advance() -> None:
        nonlocal steps
        controller.step()
        scene.step(timestep)
        mujoco.mj_step(model, data)
        steps += 1

    def still_flying() -> bool:
        return (
                steps < max_steps
                and controller.state != DroneState.GROUNDED
                and not has_diverged(drone)
        )

    if viewer:
        def step_with_overlay(handle) -> None:
            advance()
            draw_state_label(handle, drone, controller.state.name)

        aborted = run_viewer_loop(model, data, step_with_overlay, still_flying)
    else:
        aborted = False
        while still_flying():
            advance()

    return build_result(
        drone,
        target,
        controller.state,
        steps,
        timestep,
        aborted=aborted,
        diverged=has_diverged(drone),
    )


def has_diverged(drone: Drone) -> bool:
    """True once the drone is outside the world or its state has gone non-finite."""
    position = drone.sensors.pos
    return (
            not np.all(np.isfinite(position))
            or bool(np.max(np.abs(position)) > WORLD_LIMIT_M)
    )

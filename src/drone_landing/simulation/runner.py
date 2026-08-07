from pathlib import Path

import numpy as np

from ..drone import Drone
from ..enums import DroneState
from ..scene import Mission
from ..settings import DEFAULT_HOVER_ALTITUDE_M, DEFAULT_TIMEOUT_S, MODEL_PATH, WORLD_LIMIT_M
from .flight import prepare_flight
from .result import RunResult, build_result
from .viewer import draw_state_label, run_viewer_loop


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
    flight = prepare_flight(mission, model_path)
    flight.controller.take_off(hover_altitude)

    timestep = flight.timestep
    max_steps = int(timeout / timestep)
    steps = 0

    def advance() -> None:
        nonlocal steps
        flight.advance()
        steps += 1

    def still_flying() -> bool:
        return (
                steps < max_steps
                and flight.controller.state != DroneState.GROUNDED
                and not flight.controller.gave_up
                and not has_diverged(flight.drone)
        )

    if viewer:
        def step_with_overlay(handle) -> None:
            advance()
            draw_state_label(handle, flight.drone, flight.controller.state.name)

        aborted = run_viewer_loop(flight.model, flight.data, step_with_overlay, still_flying)
    else:
        aborted = False
        while still_flying():
            advance()

    return build_result(
        flight.drone,
        flight.target,
        flight.controller.state,
        steps,
        timestep,
        aborted=aborted,
        diverged=has_diverged(flight.drone),
        gave_up=flight.controller.gave_up,
    )


def has_diverged(drone: Drone) -> bool:
    """True once the drone is outside the world or its state has gone non-finite."""
    position = drone.sensors.pos
    return (
            not np.all(np.isfinite(position))
            or bool(np.max(np.abs(position)) > WORLD_LIMIT_M)
    )

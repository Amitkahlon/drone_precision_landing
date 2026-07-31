"""Fly the drone out and back along each of the four compass directions.

    python demos/fly_directions.py

Takes off, then cycles forever: tilt one way for 3 s, hold position for 6 s to
settle back over the launch point, and repeat for the next direction.
"""

from drone_landing import DroneState
from drone_landing.simulation import (
    Phase,
    PhaseCycle,
    draw_state_label,
    prepare_free_flight,
    run_viewer_loop,
)

HOVER_ALTITUDE_M = 3.0
TILT_MAGNITUDE = 0.4
TILT_DURATION_S = 3.0
RETURN_DURATION_S = 6.0

_DIRECTIONS = (
    (0, "fly forward"),
    (90, "fly right"),
    (180, "fly backward"),
    (270, "fly left"),
)


def main() -> None:
    flight = prepare_free_flight()
    controller = flight.controller
    controller.take_off(HOVER_ALTITUDE_M)

    phases = []
    for angle_deg, label in _DIRECTIONS:
        phases.append(Phase(
            action=lambda angle=angle_deg: controller.fly(angle, TILT_MAGNITUDE),
            label=label,
            duration_s=TILT_DURATION_S,
        ))
        phases.append(Phase(
            action=controller.step,
            label="return center",
            duration_s=RETURN_DURATION_S,
        ))
    cycle = PhaseCycle(phases)

    def step(handle) -> None:
        if controller.state != DroneState.HOVERING:
            controller.step()
            draw_state_label(handle, flight.drone, controller.state.name)
        else:
            draw_state_label(handle, flight.drone, cycle.run_current())
        flight.integrate()

    run_viewer_loop(flight.model, flight.data, step)


if __name__ == "__main__":
    main()

"""Yaw the drone left and right on the spot.

    python demos/yaw_rotation.py

Takes off, then cycles forever: yaw one way for 3 s, hold heading for 2 s, and
repeat in the other direction. Watch the red nose marker to see the heading.
"""

from drone_landing import DroneState
from drone_landing.simulation import (
    Phase,
    PhaseCycle,
    draw_state_label,
    ensure_viewer_thread,
    prepare_free_flight,
    run_viewer_loop,
)

HOVER_ALTITUDE_M = 3.0
YAW_RATE_RAD_S = 0.8
TURN_DURATION_S = 3.0
STOP_DURATION_S = 2.0


def main() -> None:
    ensure_viewer_thread()

    flight = prepare_free_flight()
    controller = flight.controller
    controller.take_off(HOVER_ALTITUDE_M)

    def turn(rate: float) -> Phase:
        label = "rotate left" if rate < 0 else "rotate right"
        return Phase(
            action=lambda: controller.rotate(rate),
            label=label,
            duration_s=TURN_DURATION_S,
        )

    def stop() -> Phase:
        return Phase(
            action=lambda: controller.rotate(0.0),
            label="stop",
            duration_s=STOP_DURATION_S,
        )

    cycle = PhaseCycle([
        turn(-YAW_RATE_RAD_S),
        stop(),
        turn(YAW_RATE_RAD_S),
        stop(),
    ])

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

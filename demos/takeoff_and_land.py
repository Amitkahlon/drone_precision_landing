"""Take off, hold a hover, then land back on the spot.

    python demos/takeoff_and_land.py

There is no platform here, so this exercises the controller's own takeoff,
hover and descent without any target tracking.
"""

import time

from drone_landing import DroneState
from drone_landing.simulation import (
    draw_state_label,
    ensure_viewer_thread,
    prepare_free_flight,
    run_viewer_loop,
)

HOVER_ALTITUDE_M = 3.0
HOVER_DURATION_S = 3.0


def main() -> None:
    ensure_viewer_thread()

    flight = prepare_free_flight()
    controller = flight.controller
    controller.take_off(HOVER_ALTITUDE_M)

    hover_started_at: float | None = None
    landing_triggered = False

    def step(handle) -> None:
        nonlocal hover_started_at, landing_triggered

        if controller.state == DroneState.GROUNDED:
            draw_state_label(handle, flight.drone, "grounded")
        elif controller.state != DroneState.HOVERING:
            controller.step()
            draw_state_label(handle, flight.drone, controller.state.name)
        else:
            if hover_started_at is None:
                hover_started_at = time.time()
            if not landing_triggered and time.time() - hover_started_at >= HOVER_DURATION_S:
                controller.land()
                landing_triggered = True

            controller.step()
            draw_state_label(
                handle,
                flight.drone,
                "landing" if landing_triggered else "hovering",
            )

        flight.integrate()

    run_viewer_loop(flight.model, flight.data, step)


if __name__ == "__main__":
    main()

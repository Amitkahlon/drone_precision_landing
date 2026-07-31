"""Controller behaviour, exercised against a real MuJoCo scene but no viewer."""

import numpy as np
import pytest

from drone_landing import DroneState, Mission, Platform
from drone_landing.control.gains import MAX_TILT
from drone_landing.simulation import prepare_flight, prepare_free_flight

HOVER_ALTITUDE_M = 3.0
_STEP_LIMIT = 40_000


def advance_until(flight, predicate, limit: int = _STEP_LIMIT) -> bool:
    """Step the simulation until `predicate` holds, reporting whether it did."""
    for _ in range(limit):
        if predicate():
            return True
        flight.advance()
    return predicate()


def tracking_mission() -> Mission:
    mission = Mission(start=(0, 0, 0), platform=Platform())
    mission.add_checkpoint((2.0, 0.0, 0.5), speed=0.5)
    mission.add_checkpoint((-2.0, 0.0, 0.5), speed=0.5)
    return mission


def test_starts_grounded_with_motors_off():
    flight = prepare_free_flight()

    flight.controller.step()

    assert flight.controller.state is DroneState.GROUNDED
    assert flight.drone.motors.fl == 0.0
    assert flight.drone.motors.fr == 0.0
    assert flight.drone.motors.br == 0.0
    assert flight.drone.motors.bl == 0.0


def test_take_off_enters_taking_off_then_hovers_without_a_target():
    flight = prepare_free_flight()
    flight.controller.take_off(HOVER_ALTITUDE_M)
    assert flight.controller.state is DroneState.TAKING_OFF

    reached = advance_until(
        flight, lambda: flight.controller.state is DroneState.HOVERING
    )

    assert reached
    assert flight.drone.get_z() == pytest.approx(HOVER_ALTITUDE_M, abs=0.2)


def test_take_off_enters_tracking_when_a_target_is_set():
    flight = prepare_flight(tracking_mission())
    flight.controller.take_off(HOVER_ALTITUDE_M)

    reached = advance_until(
        flight,
        lambda: flight.controller.state in (DroneState.TRACKING, DroneState.LANDING),
    )

    assert reached


def test_full_landing_sequence_ends_grounded_on_the_platform():
    flight = prepare_flight(tracking_mission())
    flight.controller.take_off(HOVER_ALTITUDE_M)

    landed = advance_until(
        flight, lambda: flight.controller.state is DroneState.GROUNDED
    )

    assert landed
    horizontal_error = np.linalg.norm(
        np.array([flight.drone.get_x(), flight.drone.get_y()])
        - flight.target.position[:2]
    )
    assert horizontal_error <= flight.target.platform.width / 2


def test_landing_cuts_the_motors_on_touchdown():
    flight = prepare_flight(tracking_mission())
    flight.controller.take_off(HOVER_ALTITUDE_M)
    advance_until(flight, lambda: flight.controller.state is DroneState.GROUNDED)

    flight.controller.step()

    assert flight.drone.motors.fl == 0.0
    assert flight.drone.motors.bl == 0.0


def test_hover_latches_the_current_position_as_the_setpoint():
    flight = prepare_free_flight(start=(1.5, -2.5, 1.0))

    flight.controller.hover()

    assert flight.controller.state is DroneState.HOVERING
    assert flight.controller._target_x == pytest.approx(1.5)
    assert flight.controller._target_y == pytest.approx(-2.5)


def test_land_command_targets_the_floor():
    flight = prepare_free_flight()

    flight.controller.land()

    assert flight.controller.state is DroneState.LANDING
    assert flight.controller._target_z == 0.0


def test_track_command_sets_tracking_state():
    flight = prepare_flight(tracking_mission())

    flight.controller.track()

    assert flight.controller.state is DroneState.TRACKING


@pytest.mark.parametrize("angle_deg", [0, 90, 180, 270])
def test_fly_tilts_within_the_authority_limit(angle_deg):
    flight = prepare_free_flight()

    flight.controller.fly(angle_deg, magnitude=1.0)

    assert abs(flight.controller._target_roll) <= MAX_TILT + 1e-12
    assert abs(flight.controller._target_pitch) <= MAX_TILT + 1e-12


def test_fly_magnitude_is_clamped_to_unit_range():
    flight = prepare_free_flight()

    flight.controller.fly(0, magnitude=5.0)
    saturated_pitch = flight.controller._target_pitch

    flight.controller.fly(0, magnitude=1.0)

    assert saturated_pitch == pytest.approx(flight.controller._target_pitch)


def test_motor_thrusts_never_leave_the_actuator_range():
    flight = prepare_flight(tracking_mission())
    flight.controller.take_off(HOVER_ALTITUDE_M)
    ceiling = flight.drone.max_thrust

    for _ in range(4000):
        flight.advance()
        for thrust in (
                flight.drone.motors.fl,
                flight.drone.motors.fr,
                flight.drone.motors.br,
                flight.drone.motors.bl,
        ):
            assert 0.0 <= thrust <= ceiling


def test_hover_thrust_supports_the_airframe():
    """Four motors at hover thrust should balance the drone's weight.

    The airframe masses declared in drone.xml sum to 0.505 kg: a 0.3 chassis,
    two 0.05 arms, four 0.01 motor columns, four 0.015 rotors and a 0.005 marker.
    """
    flight = prepare_free_flight()

    total = 4 * flight.drone.compute_hover_thrust()

    assert total == pytest.approx(0.505 * 9.81, rel=1e-6)

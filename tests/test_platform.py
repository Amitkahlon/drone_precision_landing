import numpy as np
import pytest

from drone_landing import Mission, MovingPlatform, Platform, Waypoint


def line_platform(speed: float = 1.0) -> MovingPlatform:
    """A platform shuttling between (0, 0, 0) and (10, 0, 0)."""
    return MovingPlatform(
        Platform(),
        [Waypoint((0.0, 0.0, 0.0), speed), Waypoint((10.0, 0.0, 0.0), speed)],
    )


def test_requires_at_least_two_waypoints():
    with pytest.raises(ValueError, match="at least 2 waypoints"):
        MovingPlatform(Platform(), [Waypoint((0.0, 0.0, 0.0), 1.0)])


def test_starts_at_first_waypoint():
    platform = line_platform()

    assert np.allclose(platform.position, [0.0, 0.0, 0.0])


def test_step_moves_at_the_leg_speed():
    platform = line_platform(speed=2.0)

    platform.step(0.5)

    assert np.allclose(platform.position, [1.0, 0.0, 0.0])


def test_position_is_a_copy_callers_cannot_mutate():
    platform = line_platform()

    platform.position[0] = 999.0

    assert np.allclose(platform.position, [0.0, 0.0, 0.0])


def test_velocity_points_along_the_current_leg():
    platform = line_platform(speed=3.0)

    assert np.allclose(platform.velocity, [3.0, 0.0, 0.0])


def test_reaching_a_waypoint_snaps_and_reverses():
    platform = line_platform(speed=1.0)

    platform.step(20.0)  # far past the second waypoint
    assert np.allclose(platform.position, [10.0, 0.0, 0.0])

    # The route is cyclic, so the next leg heads back to the start.
    assert np.allclose(platform.velocity, [-1.0, 0.0, 0.0])


def test_cycle_returns_to_the_start():
    platform = line_platform(speed=1.0)

    platform.step(20.0)  # snaps onto the far waypoint
    platform.step(20.0)  # and back onto the first

    assert np.allclose(platform.position, [0.0, 0.0, 0.0])
    assert np.allclose(platform.velocity, [1.0, 0.0, 0.0])


def test_many_small_steps_traverse_the_leg():
    platform = line_platform(speed=1.0)

    for _ in range(50):
        platform.step(0.1)

    assert platform.position[0] == pytest.approx(5.0)


def test_stationary_platform_reports_zero_velocity():
    platform = MovingPlatform(
        Platform(),
        [Waypoint((1.0, 1.0, 0.0), 1.0), Waypoint((1.0, 1.0, 0.0), 1.0)],
    )

    assert np.allclose(platform.velocity, [0.0, 0.0, 0.0])


def test_mission_needs_two_checkpoints_to_build_a_platform():
    mission = Mission(start=(0, 0, 0), platform=Platform())
    mission.add_checkpoint((1.0, 1.0, 0.5), speed=1.0)

    with pytest.raises(ValueError, match="at least 2 checkpoints"):
        mission.build_moving_platform()


def test_mission_builds_a_platform_from_its_checkpoints():
    mission = Mission(start=(0, 0, 0), platform=Platform(width=2.0))
    mission.add_checkpoint((1.0, 0.0, 0.5), speed=0.8)
    mission.add_checkpoint((4.0, 0.0, 0.5), speed=1.6)

    platform = mission.build_moving_platform()

    assert platform.platform.width == 2.0
    assert [wp.speed for wp in platform.waypoints] == [0.8, 1.6]
    assert np.allclose(platform.position, [1.0, 0.0, 0.5])


def test_add_checkpoint_chains():
    mission = Mission(start=(0, 0, 0), platform=Platform())

    result = mission.add_checkpoint((1.0, 1.0, 0.5), speed=1.0)

    assert result is mission

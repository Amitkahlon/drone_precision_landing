import numpy as np
import pytest

from drone_landing import ScenarioConfig, generate_scenario


def test_same_seed_gives_identical_waypoints():
    first = generate_scenario(ScenarioConfig(), seed=99)
    second = generate_scenario(ScenarioConfig(), seed=99)

    assert len(first.waypoints) == len(second.waypoints)
    for left, right in zip(first.waypoints, second.waypoints):
        assert np.array_equal(left.position, right.position)
        assert left.speed == right.speed


def test_different_seeds_give_different_waypoints():
    first = generate_scenario(ScenarioConfig(), seed=1)
    second = generate_scenario(ScenarioConfig(), seed=2)

    assert [wp.position.tolist() for wp in first.waypoints] != [
        wp.position.tolist() for wp in second.waypoints
    ]


@pytest.mark.parametrize("seed", range(15))
def test_waypoint_count_respects_configured_range(seed):
    config = ScenarioConfig(min_waypoints=4, max_waypoints=6)
    scenario = generate_scenario(config, seed)

    assert 4 <= len(scenario.waypoints) <= 6


@pytest.mark.parametrize("seed", range(15))
def test_waypoints_stay_inside_bounds(seed):
    config = ScenarioConfig(bounds_x=3.0, bounds_y=5.0, min_altitude=0.4, max_altitude=0.9)
    scenario = generate_scenario(config, seed)

    for waypoint in scenario.waypoints:
        x, y, z = waypoint.position
        assert -3.0 <= x <= 3.0
        assert -5.0 <= y <= 5.0
        assert 0.4 <= z <= 0.9


@pytest.mark.parametrize("seed", range(15))
def test_speeds_stay_inside_configured_range(seed):
    config = ScenarioConfig(min_speed=1.0, max_speed=1.5)
    scenario = generate_scenario(config, seed)

    for waypoint in scenario.waypoints:
        assert 1.0 <= waypoint.speed <= 1.5


@pytest.mark.parametrize("seed", range(15))
def test_waypoints_honour_minimum_spacing(seed):
    config = ScenarioConfig(bounds_x=8.0, bounds_y=8.0, min_spacing=2.0)
    scenario = generate_scenario(config, seed)

    positions = [wp.position for wp in scenario.waypoints]
    for i, left in enumerate(positions):
        for right in positions[i + 1:]:
            assert np.linalg.norm(left[:2] - right[:2]) >= 2.0


def test_impossible_spacing_degrades_instead_of_failing():
    """A box too tight for the requested spacing should still produce a scenario."""
    config = ScenarioConfig(
        bounds_x=0.5,
        bounds_y=0.5,
        min_spacing=50.0,
        min_waypoints=5,
        max_waypoints=5,
    )
    scenario = generate_scenario(config, seed=3)

    assert len(scenario.waypoints) == 5
    for waypoint in scenario.waypoints:
        assert np.all(np.isfinite(waypoint.position))


def test_legs_close_the_cycle():
    scenario = generate_scenario(ScenarioConfig(), seed=42)
    legs = scenario.legs

    assert len(legs) == len(scenario.waypoints)
    assert legs[-1]["to"] == 0
    for index, leg in enumerate(legs):
        assert leg["from"] == index
        assert leg["speed"] == scenario.waypoints[index].speed
        assert leg["length"] > 0


def test_to_dict_is_json_ready():
    scenario = generate_scenario(ScenarioConfig(), seed=7)
    payload = scenario.to_dict()

    assert payload["seed"] == 7
    assert payload["num_waypoints"] == len(scenario.waypoints)
    assert all(isinstance(wp["position"], list) for wp in payload["waypoints"])

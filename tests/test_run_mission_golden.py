"""Regression guard on end-to-end flight results.

These values were captured from the codebase before it was restructured, then
recaptured twice as flight behaviour deliberately changed:

- Seed 2002 used to touch down 2.47 m off the platform, and now abandons its
  descent when the platform turns away and lines up again before landing on it.
- All of them shifted slightly when the climb gained a lateral position hold.
- Seed 1068 was added with that same change. Its platform sweeps over the launch
  point, and the drone used to be bulldozed and left inverted on the floor
  without ever completing its climb; it now waits for the pad and lands.

The simulation is fully deterministic for a given seed, so any drift here means
a change altered flight behaviour rather than just moving code around.
"""

import pytest

from drone_landing import ScenarioConfig, generate_scenario, run_mission

GOLDEN_RESULTS = {
    42: {
        "landed": True,
        "on_platform": True,
        "success": True,
        "failure_reason": None,
        "final_state": "GROUNDED",
        "duration_s": 12.918,
        "steps": 6459,
        "xy_error_m": 0.0769,
        "drone_position": [-2.9315, -3.063, 0.6194],
        "platform_position": [-2.9702, -2.9966, 0.5],
    },
    1068: {
        "landed": True,
        "on_platform": True,
        "success": True,
        "failure_reason": None,
        "final_state": "GROUNDED",
        "duration_s": 11.326,
        "steps": 5663,
        "xy_error_m": 0.0902,
        "drone_position": [3.3017, -5.7733, 0.6194],
        "platform_position": [3.2161, -5.7448, 0.5],
    },
    2002: {
        "landed": True,
        "on_platform": True,
        "success": True,
        "failure_reason": None,
        "final_state": "GROUNDED",
        "duration_s": 20.63,
        "steps": 10315,
        "xy_error_m": 0.0613,
        "drone_position": [1.8622, -0.6366, 0.6192],
        "platform_position": [1.9147, -0.605, 0.5],
    },
    7: {
        "landed": True,
        "on_platform": True,
        "success": True,
        "failure_reason": None,
        "final_state": "GROUNDED",
        "duration_s": 12.01,
        "steps": 6005,
        "xy_error_m": 0.0858,
        "drone_position": [2.631, -4.4411, 0.619],
        "platform_position": [2.6151, -4.3567, 0.5],
    },
}


@pytest.mark.parametrize("seed", sorted(GOLDEN_RESULTS))
def test_run_mission_matches_the_recorded_baseline(seed):
    scenario = generate_scenario(ScenarioConfig(), seed)

    result = run_mission(scenario.build_mission()).to_dict()

    assert result == GOLDEN_RESULTS[seed]


def test_a_repeated_run_is_bit_identical():
    scenario = generate_scenario(ScenarioConfig(), 123)

    first = run_mission(scenario.build_mission()).to_dict()
    second = run_mission(scenario.build_mission()).to_dict()

    assert first == second


def test_timeout_abandons_the_run_without_landing():
    scenario = generate_scenario(ScenarioConfig(), 42)

    result = run_mission(scenario.build_mission(), timeout=1.0)

    assert not result.landed
    assert result.failure_reason == "timeout"
    assert result.duration_s == pytest.approx(1.0, abs=0.01)

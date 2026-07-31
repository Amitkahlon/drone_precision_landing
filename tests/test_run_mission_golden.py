"""Regression guard on end-to-end flight results.

These values were captured from the codebase before it was restructured. The
simulation is fully deterministic for a given seed, so any drift here means a
refactor changed flight behaviour rather than just moving code around.
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
        "duration_s": 12.94,
        "steps": 6470,
        "xy_error_m": 0.0777,
        "drone_position": [-2.9218, -3.0526, 0.6188],
        "platform_position": [-2.9611, -2.9855, 0.5],
    },
    2002: {
        "landed": True,
        "on_platform": False,
        "success": False,
        "failure_reason": "off_platform",
        "final_state": "GROUNDED",
        "duration_s": 10.7,
        "steps": 5350,
        "xy_error_m": 2.4681,
        "drone_position": [4.2459, -6.3607, 0.6186],
        "platform_position": [1.7779, -6.3862, 0.5],
    },
    7: {
        "landed": True,
        "on_platform": True,
        "success": True,
        "failure_reason": None,
        "final_state": "GROUNDED",
        "duration_s": 12.026,
        "steps": 6013,
        "xy_error_m": 0.0855,
        "drone_position": [2.6217, -4.4401, 0.619],
        "platform_position": [2.6057, -4.3562, 0.5],
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

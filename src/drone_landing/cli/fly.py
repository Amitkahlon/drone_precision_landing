"""Watch a single scenario in the MuJoCo viewer.

    drone-fly                      # the straight-line reference mission
    drone-fly square               # the square-circuit reference mission
    drone-fly random               # a fresh random scenario
    drone-fly random --seed 12345  # replay seed 12345
"""

import argparse
import random

from ..missions import MISSIONS, build_mission
from ..scenarios import ScenarioConfig, Scenario, generate_scenario
from ..settings import DEFAULT_HOVER_ALTITUDE_M
from ..simulation import (
    draw_state_label,
    ensure_viewer_thread,
    prepare_flight,
    run_mission,
    run_viewer_loop,
)

_RANDOM = "random"
_DEFAULT_SCENARIO = "straight-line"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    # Every path here opens a viewer, so hand over to mjpython before doing work.
    ensure_viewer_thread()

    if args.scenario == _RANDOM:
        return _fly_random(args.seed)
    return _fly_reference_mission(args.scenario, args.hover_altitude)


def _fly_reference_mission(name: str, hover_altitude: float) -> int:
    """Fly a hand-built mission, leaving the viewer open once it has settled."""
    flight = prepare_flight(build_mission(name))
    flight.controller.take_off(hover_altitude)

    def step(handle) -> None:
        flight.advance()
        draw_state_label(handle, flight.drone, flight.controller.state.name)

    run_viewer_loop(flight.model, flight.data, step)
    return 0


def _fly_random(seed: int | None) -> int:
    """Fly a generated scenario, closing the viewer as soon as it touches down."""
    if seed is None:
        seed = random.SystemRandom().randrange(2 ** 32)
    scenario = generate_scenario(ScenarioConfig(), seed)
    _print_scenario(scenario)

    result = run_mission(scenario.build_mission(), viewer=True)
    print(
        f"\n{'landed on platform' if result.success else result.failure_reason} "
        f"after {result.duration_s:.1f}s, XY error {result.xy_error_m:.3f} m"
    )
    return 0


def _print_scenario(scenario: Scenario) -> None:
    print(f"seed {scenario.seed}, {len(scenario.waypoints)} waypoints")
    for index, waypoint in enumerate(scenario.waypoints):
        x, y, z = waypoint.position
        print(f"  {index}: ({x:6.2f}, {y:6.2f}, {z:5.2f})  speed {waypoint.speed:.2f} m/s")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "scenario",
        nargs="?",
        default=_DEFAULT_SCENARIO,
        choices=sorted(MISSIONS) + [_RANDOM],
        help="a reference mission by name, or 'random' for a generated scenario",
    )
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for the random scenario; ignored otherwise")
    parser.add_argument("--hover-altitude", type=float, default=DEFAULT_HOVER_ALTITUDE_M)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())

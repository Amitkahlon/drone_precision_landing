"""A single randomized scenario in the viewer.

    drone-fly           # new random scenario
    drone-fly 12345     # replay seed 12345
"""

import random
import sys

from ..scenarios import ScenarioConfig, generate_scenario
from ..simulation import run_mission


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    seed = int(argv[0]) if argv else random.SystemRandom().randrange(2 ** 32)
    scenario = generate_scenario(ScenarioConfig(), seed)

    print(f"seed {seed}, {len(scenario.waypoints)} waypoints")
    for i, waypoint in enumerate(scenario.waypoints):
        x, y, z = waypoint.position
        print(f"  {i}: ({x:6.2f}, {y:6.2f}, {z:5.2f})  speed {waypoint.speed:.2f} m/s")

    result = run_mission(scenario.build_mission(), viewer=True)

    print(
        f"\n{'landed on platform' if result.success else result.failure_reason} "
        f"after {result.duration_s:.1f}s, XY error {result.xy_error_m:.3f} m"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

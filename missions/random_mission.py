"""A single randomized scenario in the viewer.

    python missions/random_mission.py           # new random scenario
    python missions/random_mission.py 12345     # replay seed 12345
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from controls import ScenarioConfig, generate_scenario, run_mission

seed = int(sys.argv[1]) if len(sys.argv) > 1 else random.SystemRandom().randrange(2 ** 32)
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

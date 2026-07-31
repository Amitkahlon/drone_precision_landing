# Drone Precision Landing

A [MuJoCo](https://mujoco.readthedocs.io/) simulation of a quadrotor that takes off, tracks a moving
landing platform, and lands on it. Scenarios can be flown one at a time in the viewer or in headless
batches to measure how often the controller succeeds.

## Setup

```bash
source .venv/bin/activate
pip install -e ".[dev]"
```

The editable install is what puts `drone_landing` on the import path and creates the `drone-fly` and
`drone-batch` commands. Runtime dependencies are pinned in `requirements.txt` (`mujoco`, `numpy`,
`glfw`, `PyOpenGL`); `requirements-dev.txt` adds `pytest`.

## Layout

```
src/drone_landing/
  settings.py      paths, timeouts and world limits
  enums.py         Motor, Sensor, DroneState
  assets/          the MJCF quadrotor model
  drone/           Drone, plus named motor and sensor accessors
  control/         the flight state machine, its PD loops and its gains
  scene/           Mission, Platform, MovingPlatform and the MJCF scene builder
  scenarios/       ScenarioConfig and the seeded random generator
  simulation/      flight assembly, the run loop, results and viewer plumbing
  missions/        hand-built reference missions
  batch/           per-run records, batch summaries and their report
  cli/             the drone-fly and drone-batch entry points
demos/             interactive viewer scripts for individual manoeuvres
tests/             headless pytest suite
results/           batch output (gitignored)
```

The dependency direction runs one way: `cli` uses `simulation` and `batch`, `simulation` uses `scene`
and `control`, and `control` uses `drone`. Nothing lower reaches back up.

## Flying a single scenario

```bash
drone-fly                      # straight-line reference mission
drone-fly square               # square-circuit reference mission
drone-fly random               # a fresh random scenario, seed printed on start
drone-fly random --seed 12345  # replay seed 12345
```

The two reference missions leave the viewer open after the drone settles, so you can inspect the final
state; `random` closes as soon as the run resolves and prints the outcome.

On macOS, opening a MuJoCo window requires the `mjpython` launcher, because Cocoa insists on owning the
process's main thread. Any command that opens a viewer detects this and re-runs itself under `mjpython`
automatically, so the commands above work as written; the same applies to `drone-batch --viewer` and the
demo scripts. Headless runs are unaffected and use plain `python`.

A mission is a start position for the drone plus a cyclic list of platform waypoints, where each
waypoint carries the speed of the leg leaving it:

```python
from drone_landing import Mission, Platform

mission = Mission(start=(0, 0, 0), platform=Platform(width=1.0, depth=1.0, thickness=0.05))
mission.add_checkpoint((5, 5, 0.5), speed=0.8)
mission.add_checkpoint((-5, 5, 0.5), speed=0.8)
```

## Running a batch

```bash
drone-batch --runs 50
```

Runs are roughly 40x faster than real time, so 50 scenarios take about 15 seconds. Each run appends a
JSON object to `results/runs.jsonl` relative to the working directory, so repeated batches accumulate
into one dataset. A per-batch summary is written beside it as `summary_<batch_id>.json` and printed:

```
50 runs, 39 succeeded (78.0%), 11 off-platform, 0 timed out, 0 diverged

  by waypoint count:
             3    8/10   80.0%
             4    6/8    75.0%
             ...
  by mean leg speed:
     0.50-0.88    2/2   100.0%
     0.88-1.25   19/25   76.0%
     1.25-1.62   17/22   77.3%
     1.62-2.00    1/1   100.0%
```

### Replaying a failure

Run `i` of a batch uses seed `base_seed + i`, and every record stores its own seed, so any single run
replays exactly:

```bash
drone-batch --seed 2002 --runs 1 --viewer
```

The batch summary prints this command for the first failing seed, and lists all of them under
`failed_seeds`. Omitting `--seed` draws a base seed from system entropy and prints it, so even
unseeded batches stay reproducible after the fact.

## Configuration

| Flag | Default | Meaning |
| --- | --- | --- |
| `-n`, `--runs` | `20` | scenarios in the batch |
| `--seed` | random | base RNG seed; run `i` uses `seed + i` |
| `--bounds X Y` | `8.0 8.0` | half-extents of the waypoint box around the launch point |
| `--min-waypoints` / `--max-waypoints` | `3` / `8` | waypoints per mission |
| `--min-speed` / `--max-speed` | `0.5` / `2.0` | per-leg speed range in m/s |
| `--min-altitude` / `--max-altitude` | `0.5` / `0.5` | platform altitude range; equal values keep it planar like the reference missions |
| `--min-spacing` | `2.0` | minimum XY distance between waypoints |
| `--timeout` | `120.0` | per-run simulated seconds before the run is abandoned |
| `--hover-altitude` | `3.0` | takeoff altitude |
| `--output` | `results/runs.jsonl` | results file (appended) |
| `--viewer` | off | watch each run instead of running headless |

The scenario flags read their defaults straight off `ScenarioConfig`, so the CLI and the programmatic
API cannot drift apart:

```python
from drone_landing import ScenarioConfig, generate_scenario, run_mission

scenario = generate_scenario(ScenarioConfig(max_speed=3.0), seed=42)
result = run_mission(scenario.build_mission())
print(result.success, result.xy_error_m)
```

Paths, timeouts and the world limit live in `src/drone_landing/settings.py`; the controller's gains and
thresholds live in `src/drone_landing/control/gains.py`.

## Landing outcomes

`DroneController` declares touchdown when the drone drops within 0.12 m of the platform's altitude.
That check is Z-only, so a drone that drifts off during descent still reports `GROUNDED`. Each record
therefore keeps the raw controller verdict and the stricter one separately:

- `landed` — the controller reached `DroneState.GROUNDED`, its own criterion.
- `on_platform` — touchdown happened inside the platform footprint.
- `success` — both of the above.
- `failure_reason` — `off_platform`, `timeout` (never touched down within `--timeout`), `diverged`
  (the drone left the 20x20 m world), `aborted` (viewer closed), or `null` on success.

Alongside those, every record carries the run id, seed, timestamp, the generated waypoints with their
per-leg speeds and lengths, min/mean/max leg speed, mission duration, step count, final XY error, the
final drone and platform positions, and the full config used to generate it.

## Demos

Interactive scripts that exercise one controller capability each, without a landing target:

```bash
python demos/takeoff_and_land.py   # climb, hold a hover, descend
python demos/fly_directions.py     # tilt out and back along each compass direction
python demos/yaw_rotation.py       # yaw left and right on the spot
```

Each opens the viewer and loops until you close the window, re-running itself under `mjpython` on macOS
as described above.

## Tests

```bash
pytest
```

The suite is headless and runs in a few seconds. `tests/test_run_mission_golden.py` pins three seeds
to the exact results the simulation produced before the codebase was restructured, so a change that
alters flight behaviour fails there rather than passing silently.

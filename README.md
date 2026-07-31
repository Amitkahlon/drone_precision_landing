# Drone Precision Landing

A [MuJoCo](https://mujoco.readthedocs.io/) simulation of a quadrotor that takes off, tracks a moving
landing platform, and lands on it. The quadrotor model lives in `models/drone.xml`; the control stack
lives in `controls/`.

## Setup

```bash
source .venv/bin/activate
```

Dependencies are pinned in `requirements.txt` (`mujoco`, `numpy`, `matplotlib`, `glfw`, `PyOpenGL`).
The scenario generator and batch runner add nothing beyond the standard library.

## Hand-built scenarios

`missions/mission1.py` (2 waypoints at 0.8 m/s) and `missions/mission2.py` (a 4-waypoint square at
1.2 m/s) define the platform trajectory by hand and open the viewer. `python main.py` runs mission1.

A mission is a start position for the drone plus a cyclic list of platform waypoints, where each
waypoint carries the speed of the leg leaving it:

```python
mission = Mission(start=(0, 0, 0), platform=Platform(width=1.0, depth=1.0, thickness=0.05))
mission.add_checkpoint((5, 5, 0.5), speed=0.8)
mission.add_checkpoint((-5, 5, 0.5), speed=0.8)
```

## Randomized scenarios

`controls/scenario_generator.py` samples those same waypoints and per-leg speeds from a seeded RNG,
and `controls/simulation.py` flies any mission end to end — headless for batches, or in the viewer for
debugging. Neither the sample missions nor the controller are touched; the generator only produces
mission definitions and the runner only calls into the existing classes.

### One scenario, in the viewer

```bash
python missions/random_mission.py          # fresh random scenario, seed printed on start
python missions/random_mission.py 12345    # replay seed 12345
```

### A batch, headless

```bash
python batch_runner.py --runs 50
```

Runs are roughly 40x faster than real time, so 50 scenarios take about 15 seconds. Each run appends a
JSON object to `results/runs.jsonl`, so repeated batches accumulate into one dataset. A per-batch
summary is written to `results/summary_<batch_id>.json` and printed:

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
python batch_runner.py --seed 2002 --runs 1 --viewer
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
| `--min-altitude` / `--max-altitude` | `0.5` / `0.5` | platform altitude range; equal values keep it planar like the sample missions |
| `--min-spacing` | `2.0` | minimum XY distance between waypoints |
| `--timeout` | `120.0` | per-run simulated seconds before the run is abandoned |
| `--hover-altitude` | `3.0` | takeoff altitude |
| `--output` | `results/runs.jsonl` | results file (appended) |
| `--viewer` | off | watch each run instead of running headless |

The same surface is available programmatically as `ScenarioConfig`:

```python
from controls import ScenarioConfig, generate_scenario, run_mission

scenario = generate_scenario(ScenarioConfig(max_speed=3.0), seed=42)
result = run_mission(scenario.build_mission())
print(result.success, result.xy_error_m)
```

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

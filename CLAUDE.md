# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

All Python work runs inside the local virtual environment, with the project installed editable:

```bash
source .venv/bin/activate
pip install -e ".[dev]"
```

Key installed packages: `mujoco 3.8.1`, `numpy`, `glfw`, `PyOpenGL`, `pytest`.

## Project overview

A drone precision-landing simulation built with [MuJoCo](https://mujoco.readthedocs.io/). The
quadrotor is defined in `src/drone_landing/assets/drone.xml` and loaded through
`drone_landing.settings.MODEL_PATH`. `SceneBuilder` injects mocap platform bodies into that MJCF as a
string before compiling it, which is why platforms must be registered before `build()` and drones
after.

## Commands

```bash
drone-fly [straight-line|square|random] [--seed N]   # one scenario in the viewer
drone-batch --runs 50                                # headless batch to results/
pytest                                               # headless test suite
python demos/takeoff_and_land.py                     # single-manoeuvre demos
```

## Architecture

Layered, with dependencies pointing one way only:

- `drone/` — `Drone` wraps the MuJoCo model and data; `MotorBank` and `SensorArray` give named access
  to `data.ctrl` and `data.sensordata`.
- `control/` — `DroneController` is purely the state machine
  (`GROUNDED → TAKING_OFF → TRACKING → LANDING → GROUNDED`, with `HOVERING` when no target is set).
  Every PD loop is in `ThrustMixer`; every tuning constant is in `gains.py`.
- `scene/` — `Mission` is a scenario definition, `Platform`/`Waypoint`/`MovingPlatform` are the target,
  and `SceneBuilder` compiles them into a MuJoCo model.
- `simulation/` — `prepare_flight` assembles a `Flight`, `run_mission` drives it to an outcome, and
  `viewer.py` holds the only copy of the wall-clock pacing and the state-label overlay.
- `scenarios/`, `missions/`, `batch/`, `cli/` — scenario generation, hand-built reference routes,
  result aggregation, and entry points.

## Constraints

- The simulation is deterministic per seed. `tests/test_run_mission_golden.py` pins seeds 42, 2002 and
  7 to exact results; run `pytest` after any change under `control/`, `drone/` or `scene/`.
- Preserve floating-point expressions verbatim when refactoring control code. Algebraically equivalent
  rewrites (for example `np.hypot` in place of an expanded `sqrt`) can shift a state transition by a
  step and break the golden results.

## MuJoCo concepts used

- **Free joint** (`type="free"`) on the drone body gives it 6-DOF; `qpos[0:3]` = XYZ,
  `qpos[3:7]` = quaternion.
- **Site-based motors** apply force along the body-frame Z-axis, so roll and pitch torques come for
  free from the off-centre site positions; `gear[5] = ±0.01` supplies each rotor's reactive yaw torque.
- **Mocap bodies** drive the platforms, so they move on rails and ignore the drone landing on them.
- **Passive viewer** (`launch_passive`) allows manual camera control alongside the simulation loop.

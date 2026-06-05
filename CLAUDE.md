# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

All Python work runs inside the local virtual environment:

```bash
source .venv/bin/activate
python main.py
```

Key installed packages: `mujoco 3.8.1`, `numpy`, `matplotlib`, `glfw`, `PyOpenGL`.

## Project overview

A drone precision-landing simulation built with [MuJoCo](https://mujoco.readthedocs.io/). The drone model is defined as an inline MJCF XML string inside `main.py` and loaded at runtime via `mujoco.MjModel.from_xml_string()`.

## Architecture

`main.py` is the single entry point and contains three logical sections:

1. **Model definition** — `drone_xml` (MJCF string): declares the world, drone body, free joint, four motors, and a marker geom for orientation.
2. **Simulation setup** — `MjModel` + `MjData` created from the XML string.
3. **Control loop** — `mujoco.viewer.launch_passive()` drives the loop; each step calls `mj_step`, reads `data.qpos[0:3]` for drone XYZ, and syncs the viewer.

### MuJoCo concepts used

- **Free joint** (`type="free"`) on the drone body gives it 6-DOF; `qpos[0:3]` = XYZ, `qpos[3:7]` = quaternion.
- **Motors** with `gear="0 0 1 0 0 0"` apply force only along the world Z-axis (thrust). Torque/attitude control requires additional gear components.
- **Passive viewer** (`launch_passive`) allows manual control alongside the simulation loop.

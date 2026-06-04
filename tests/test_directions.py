import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mujoco
import mujoco.viewer
import time
import numpy as np

from controls import Drone, DroneController, Mission, SceneBuilder, DroneState

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "drone.xml")
model = mujoco.MjModel.from_xml_path(model_path)
data  = mujoco.MjData(model)

drone      = Drone(model, data)
controller = DroneController(drone)

scene = SceneBuilder(model, data)
scene.add_drone("main", drone)
scene.apply_mission(Mission((0, 0, 0), (0, 0, 0)))

controller.take_off(3.0)

PHASES = [
    (lambda: controller.fly(  0, 0.4), "fly forward",  3.0),
    (lambda: controller.step(),         "return center", 6.0),
    (lambda: controller.fly( 90, 0.4), "fly right",     3.0),
    (lambda: controller.step(),         "return center", 6.0),
    (lambda: controller.fly(180, 0.4), "fly backward",  3.0),
    (lambda: controller.step(),         "return center", 6.0),
    (lambda: controller.fly(270, 0.4), "fly left",      3.0),
    (lambda: controller.step(),         "return center", 6.0),
]

TOTAL_CYCLE   = sum(d for _, _, d in PHASES)
PHASE_OFFSETS = []
t = 0.0
for _, _, d in PHASES:
    PHASE_OFFSETS.append(t)
    t += d

hover_start_time = None

def current_phase(elapsed: float):
    idx = 0
    for i, offset in enumerate(PHASE_OFFSETS):
        if elapsed >= offset:
            idx = i
    return PHASES[idx]

def update_label(viewer, label: str) -> None:
    viewer.user_scn.ngeom = 1
    g = viewer.user_scn.geoms[0]
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.zeros(3), np.zeros(3), np.eye(3).flatten(),
        np.array([0.0, 0.0, 0.0, 0.0]),
    )
    g.pos[:] = [drone.position.x, drone.position.y, drone.position.z + 0.4]
    g.size[:] = [0.01, 0.01, 0.01]
    g.label = label

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        if controller.state != DroneState.HOVERING:
            controller.step()
            update_label(viewer, controller.state.name)
        else:
            if hover_start_time is None:
                hover_start_time = time.time()

            elapsed = (time.time() - hover_start_time) % TOTAL_CYCLE
            action, label, _ = current_phase(elapsed)
            action()
            update_label(viewer, label)

        mujoco.mj_step(model, data)
        viewer.sync()

        step_elapsed = time.time() - step_start
        remaining = model.opt.timestep - step_elapsed
        if remaining > 0:
            time.sleep(remaining)
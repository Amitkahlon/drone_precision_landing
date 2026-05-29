import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mujoco
import mujoco.viewer
import time
import numpy as np

from controls import Drone, DroneController, Mission, SceneBuilder, DroneState

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "drone.xml")

scene       = SceneBuilder(model_path)
model, data = scene.build()

drone      = Drone(model, data)
controller = DroneController(drone)

scene.add_drone("main", drone)
scene.apply_mission(Mission((0, 0, 0), (0, 0, 0)))

controller.take_off(3.0)

HOVER_DURATION = 3.0

hover_start_time  = None
landing_triggered = False

def update_label(viewer, label: str) -> None:
    viewer.user_scn.ngeom = 1
    g = viewer.user_scn.geoms[0]
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.zeros(3), np.zeros(3), np.eye(3).flatten(),
        np.array([0.0, 0.0, 0.0, 0.0]),
    )
    g.pos[:] = [drone.get_x(), drone.get_y(), drone.get_z() + 0.4]
    g.size[:] = [0.01, 0.01, 0.01]
    g.label = label

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        if controller.state == DroneState.GROUNDED:
            update_label(viewer, "grounded")
        elif controller.state != DroneState.HOVERING:
            controller.step()
            update_label(viewer, controller.state.name)
        else:
            if hover_start_time is None:
                hover_start_time = time.time()

            elapsed = time.time() - hover_start_time
            if not landing_triggered and elapsed >= HOVER_DURATION:
                controller.land()
                landing_triggered = True

            controller.step()
            update_label(viewer, "hovering" if not landing_triggered else "landing")

        mujoco.mj_step(model, data)
        viewer.sync()

        step_elapsed = time.time() - step_start
        remaining = model.opt.timestep - step_elapsed
        if remaining > 0:
            time.sleep(remaining)
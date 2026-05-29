import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mujoco
import mujoco.viewer
import time
import numpy as np

from controls import Drone, DroneController, Mission, SceneBuilder, Platform

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "drone.xml")

mission = Mission(
    start=(0, 0, 0),
    platform=Platform(width=1.0, depth=1.0, thickness=0.05, color=(0.1, 0.5, 0.9, 1.0)),
)
mission.add_checkpoint(( 4,  4, 0.5), speed=1.2)
mission.add_checkpoint((-4,  4, 0.5), speed=1.2)
mission.add_checkpoint((-4, -4, 0.5), speed=1.2)
mission.add_checkpoint(( 4, -4, 0.5), speed=1.2)

platform = mission.build_platform()

scene = SceneBuilder(model_path)
scene.add_moving_platform("target", platform)
model, data = scene.build()

drone      = Drone(model, data)
controller = DroneController(drone)

scene.add_drone("main", drone)
scene.apply_mission(mission)

controller.set_target(platform)
controller.take_off(3.0)

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

        controller.step()
        scene.step(model.opt.timestep)

        mujoco.mj_step(model, data)
        update_label(viewer, controller.state.name)
        viewer.sync()

        step_elapsed = time.time() - step_start
        remaining = model.opt.timestep - step_elapsed
        if remaining > 0:
            time.sleep(remaining)
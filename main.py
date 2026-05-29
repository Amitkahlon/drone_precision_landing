import mujoco
import mujoco.viewer
import time
import os

from controls import Drone, DroneController, Mission, SceneBuilder

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "drone.xml")
model = mujoco.MjModel.from_xml_path(model_path)
data  = mujoco.MjData(model)

mission    = Mission((0, 0, 0), (0, 0, 0))
drone      = Drone(model, data)
controller = DroneController(drone)

scene = SceneBuilder(model, data)
scene.add_drone("main", drone)
scene.apply_mission(mission)

controller.take_off(3.0)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        controller.step()

        mujoco.mj_step(model, data)
        viewer.sync()

        step_elapsed = time.time() - step_start
        remaining = model.opt.timestep - step_elapsed
        if remaining > 0:
            time.sleep(remaining)
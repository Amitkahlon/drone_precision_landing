import mujoco
import mujoco.viewer
import time
import os

from controls import DroneController, Mission, Sensor

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "drone.xml")
model = mujoco.MjModel.from_xml_path(model_path)
data  = mujoco.MjData(model)

mission = Mission((0, 0, 2), (0, 0, 0))
controller = DroneController(model, data)
controller.set_pos_in_world(0, 0, 2)

target_z = mission.start[2]

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        controller.hover(target_z)
        mujoco.mj_step(model, data)

        viewer.sync()

        elapsed = time.time() - step_start
        remaining = model.opt.timestep - elapsed
        if remaining > 0:
            time.sleep(remaining)

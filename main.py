import mujoco
import mujoco.viewer
import time
import os

from controls import Drone, DroneController, Mission, SceneBuilder, Platform, Waypoint, MovingPlatform

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "drone.xml")

platform = MovingPlatform(
    Platform(width=1.0, depth=1.0, thickness=0.05, color=(0.9, 0.1, 0.1, 1.0)),
    [
        Waypoint((-3, 0, 0.5), speed=0.5),
        Waypoint(( 3, 0, 0.5), speed=0.5),
    ],
)

scene = SceneBuilder(model_path)
scene.add_moving_platform("target", platform)
model, data = scene.build()

drone      = Drone(model, data)
controller = DroneController(drone)

scene.add_drone("main", drone)
scene.apply_mission(Mission((0, 0, 0), (0, 0, 0)))

controller.take_off(3.0)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        controller.step()
        scene.step(model.opt.timestep)

        mujoco.mj_step(model, data)
        viewer.sync()

        step_elapsed = time.time() - step_start
        remaining = model.opt.timestep - step_elapsed
        if remaining > 0:
            time.sleep(remaining)
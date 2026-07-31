"""Assembly of a ready-to-fly scene: model, drone, controller and target."""

from dataclasses import dataclass
from pathlib import Path

import mujoco

from ..control import DroneController
from ..drone import Drone
from ..scene import Mission, MovingPlatform, SceneBuilder
from ..settings import MODEL_PATH

_TARGET_KEY = "target"
_DRONE_KEY = "main"


@dataclass
class Flight:
    """Everything a simulation loop needs to step the scene forward."""

    scene: SceneBuilder
    model: mujoco.MjModel
    data: mujoco.MjData
    drone: Drone
    controller: DroneController
    target: MovingPlatform | None

    @property
    def timestep(self) -> float:
        return self.model.opt.timestep

    def advance(self) -> None:
        """Run one control step, move the platforms, then integrate the physics."""
        self.controller.step()
        self.scene.step(self.timestep)
        mujoco.mj_step(self.model, self.data)


def prepare_flight(mission: Mission, model_path: str | Path = MODEL_PATH) -> Flight:
    """Build a scene for `mission` with the controller already aimed at its platform."""
    target = mission.build_moving_platform()

    scene = SceneBuilder(model_path)
    scene.add_moving_platform(_TARGET_KEY, target)
    flight = _assemble(scene, target)

    scene.apply_mission(mission)
    flight.controller.set_target(target)
    return flight


def prepare_free_flight(
        start: tuple[float, float, float] = (0.0, 0.0, 0.0),
        model_path: str | Path = MODEL_PATH,
) -> Flight:
    """Build a bare scene with no landing target, for manoeuvring demos."""
    scene = SceneBuilder(model_path)
    flight = _assemble(scene, target=None)
    scene.place_drone(start)
    return flight


def _assemble(scene: SceneBuilder, target: MovingPlatform | None) -> Flight:
    """Compile the scene, then attach a drone and controller to the built model."""
    model, data = scene.build()
    drone = Drone(model, data)
    scene.add_drone(_DRONE_KEY, drone)
    return Flight(
        scene=scene,
        model=model,
        data=data,
        drone=drone,
        controller=DroneController(drone),
        target=target,
    )

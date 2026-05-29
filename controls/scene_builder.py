import mujoco

from .mission import Mission
from .drone import Drone


class SceneBuilder:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model   = model
        self.data    = data
        self._drones: dict[str, Drone] = {}

    def add_drone(self, key: str, drone: Drone) -> None:
        self._drones[key] = drone

    def get_drone(self, key: str) -> Drone:
        return self._drones[key]

    def place_drone(self, drone_key: str, x: float, y: float, z: float) -> None:
        drone = self._drones[drone_key]
        adr = drone.qpos_adr
        self.data.qpos[adr]           = x
        self.data.qpos[adr + 1]       = y
        self.data.qpos[adr + 2]       = z

        # its about its tilt, need to figure it out:
        self.data.qpos[adr + 3 : adr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)

    def apply_mission(self, mission: Mission) -> None:
        # TODO: assuming we have only one drone atm
        key = next(iter(self._drones))
        self.place_drone(key, *mission.start)




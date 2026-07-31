from pathlib import Path

import mujoco

from ..drone import Drone
from .mission import Mission
from .platform import MovingPlatform


class SceneBuilder:
    """Assembles a MuJoCo scene from the drone model plus generated platform bodies.

    Platforms have to be registered before `build`, because they are injected into
    the MJCF as extra bodies; drones are registered after, because constructing a
    Drone requires the compiled model.
    """

    def __init__(self, xml_path: str | Path):
        self._xml_path = xml_path
        self._drones: dict[str, Drone] = {}
        self._moving_platforms: dict[str, MovingPlatform] = {}
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None

    # --- pre-build registration ---

    def add_moving_platform(self, key: str, platform: MovingPlatform) -> None:
        self._moving_platforms[key] = platform

    def build(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        self.model = mujoco.MjModel.from_xml_string(self._build_xml())
        self.data = mujoco.MjData(self.model)
        self._bind_platforms()
        return self.model, self.data

    # --- post-build scene setup ---

    def add_drone(self, key: str, drone: Drone) -> None:
        self._drones[key] = drone

    def place_drone(self, position) -> None:
        """Move the drone to `position`.

        A scene describes one launch point, so only the first registered drone is
        placed; it has never held more than one.
        """
        if not self._drones:
            raise ValueError("no drone registered; call add_drone after build")
        self._place_drone(next(iter(self._drones.values())), *position)

    def apply_mission(self, mission: Mission) -> None:
        self.place_drone(mission.start)

    # --- per-step update ---

    def step(self, dt: float) -> None:
        for platform in self._moving_platforms.values():
            platform.step(dt)

    # --- internals ---

    def _build_xml(self) -> str:
        with open(self._xml_path) as f:
            xml = f.read()
        fragments = [
            self._platform_xml(key, platform)
            for key, platform in self._moving_platforms.items()
        ]
        if fragments:
            injection = '\n    '.join(fragments)
            xml = xml.replace('</worldbody>', f'    {injection}\n  </worldbody>')
        return xml

    def _platform_xml(self, key: str, moving_platform: MovingPlatform) -> str:
        """A mocap body for the platform, so it is positioned rather than simulated."""
        platform = moving_platform.platform
        red, green, blue, alpha = platform.color
        x, y, z = moving_platform.waypoints[0].position
        return (
            f'<body name="platform_{key}" mocap="true" '
            f'pos="{x} {y} {z}">'
            f'<geom type="box" size="{platform.width / 2} {platform.depth / 2} '
            f'{platform.thickness / 2}" '
            f'rgba="{red} {green} {blue} {alpha}"/>'
            f'</body>'
        )

    def _bind_platforms(self) -> None:
        for key, platform in self._moving_platforms.items():
            body_id = self.model.body(f"platform_{key}").id
            platform.bind(self.model.body_mocapid[body_id], self.data)

    def _place_drone(self, drone: Drone, x: float, y: float, z: float) -> None:
        address = drone.qpos_adr
        self.data.qpos[address:address + 3] = [x, y, z]
        self.data.qpos[address + 3:address + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)

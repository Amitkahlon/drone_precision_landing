import mujoco

from .drone import Drone
from .mission import Mission
from .moving_platform import MovingPlatform


class SceneBuilder:
    def __init__(self, xml_path: str):
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

    def apply_mission(self, mission: Mission) -> None:
        first = next(iter(self._drones.values()))
        self._place_drone(first, *mission.start)

    # --- per-step update ---

    def step(self, dt: float) -> None:
        for platform in self._moving_platforms.values():
            platform.step(dt)

    # --- internals ---

    def _build_xml(self) -> str:
        with open(self._xml_path) as f:
            xml = f.read()
        fragments = [self._platform_xml(key, mp) for key, mp in self._moving_platforms.items()]
        if fragments:
            injection = '\n    '.join(fragments)
            xml = xml.replace('</worldbody>', f'    {injection}\n  </worldbody>')
        return xml

    def _platform_xml(self, key: str, mp: MovingPlatform) -> str:
        p = mp.platform
        r, g, b, a = p.color
        pos = mp.waypoints[0].position
        return (
            f'<body name="platform_{key}" mocap="true" '
            f'pos="{pos[0]} {pos[1]} {pos[2]}">'
            f'<geom type="box" size="{p.width / 2} {p.depth / 2} {p.thickness / 2}" '
            f'rgba="{r} {g} {b} {a}"/>'
            f'</body>'
        )

    def _bind_platforms(self) -> None:
        for key, mp in self._moving_platforms.items():
            body_id = self.model.body(f"platform_{key}").id
            mocap_id = self.model.body_mocapid[body_id]
            mp._bind(mocap_id, self.data)

    def _place_drone(self, drone: Drone, x: float, y: float, z: float) -> None:
        adr = drone.qpos_adr
        self.data.qpos[adr:adr + 3] = [x, y, z]
        self.data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)

import mujoco

from .motors import MotorBank
from .sensors import SensorArray


class Drone:
    """The simulated quadrotor: its motors, its sensors and its pose."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self._model = model
        self._data = data
        self._body_id = model.body("drone").id
        joint_id = model.body_jntadr[self._body_id]
        self.qpos_adr = int(model.jnt_qposadr[joint_id])
        self.motors = MotorBank(data.ctrl)
        self.sensors = SensorArray(model, data)

    def get_x(self) -> float: return float(self.sensors.pos[0])

    def get_y(self) -> float: return float(self.sensors.pos[1])

    def get_z(self) -> float: return float(self.sensors.pos[2])

    def compute_hover_thrust(self) -> float:
        """Per-motor thrust that exactly cancels gravity, i.e. the control mid-point."""
        total_mass = self._model.body_mass[self._body_id]
        gravity = abs(self._model.opt.gravity[2])
        return (total_mass * gravity) / 4.0

    @property
    def max_thrust(self) -> float:
        return float(self._model.actuator_ctrlrange[0, 1])

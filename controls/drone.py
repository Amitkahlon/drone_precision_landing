import mujoco
import numpy as np

from .enums import Motor, Sensor


class _MotorAccessor:
    """Read/write individual motor thrusts via drone.motors.<name>."""

    def __init__(self, ctrl):
        object.__setattr__(self, '_ctrl', ctrl)

    @property
    def fl(self) -> float: return float(self._ctrl[Motor.FL])

    @fl.setter
    def fl(self, v: float): self._ctrl[Motor.FL] = v

    @property
    def fr(self) -> float: return float(self._ctrl[Motor.FR])

    @fr.setter
    def fr(self, v: float): self._ctrl[Motor.FR] = v

    @property
    def br(self) -> float: return float(self._ctrl[Motor.BR])

    @br.setter
    def br(self, v: float): self._ctrl[Motor.BR] = v

    @property
    def bl(self) -> float: return float(self._ctrl[Motor.BL])

    @bl.setter
    def bl(self, v: float): self._ctrl[Motor.BL] = v


class _SensorAccessor:
    """Read sensor arrays via drone.sensors.<name>."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        object.__setattr__(self, '_model', model)
        object.__setattr__(self, '_data', data)

    def _read(self, sensor: Sensor) -> np.ndarray:
        sid = self._model.sensor(sensor).id
        adr = self._model.sensor_adr[sid]
        return self._data.sensordata[adr:adr + self._model.sensor_dim[sid]]

    @property
    def accel(self) -> np.ndarray: return self._read(Sensor.ACCEL)

    @property
    def gyro(self) -> np.ndarray: return self._read(Sensor.GYRO)

    @property
    def pos(self) -> np.ndarray: return self._read(Sensor.POS)

    @property
    def quat(self) -> np.ndarray: return self._read(Sensor.QUAT)

    @property
    def linvel(self) -> np.ndarray: return self._read(Sensor.LINVEL)

    @property
    def angvel(self) -> np.ndarray: return self._read(Sensor.ANGVEL)


class Drone:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self._model = model
        self._data = data
        self._body_id = model.body("drone").id
        joint_id = model.body_jntadr[self._body_id]
        self.qpos_adr = int(model.jnt_qposadr[joint_id])
        self.motors = _MotorAccessor(data.ctrl)
        self.sensors = _SensorAccessor(model, data)

    def get_x(self) -> float: return float(self.sensors.pos[0])

    def get_y(self) -> float: return float(self.sensors.pos[1])

    def get_z(self) -> float: return float(self.sensors.pos[2])

    def compute_hover_thrust(self) -> float:
        total_mass = self._model.body_mass[self._body_id]
        g = abs(self._model.opt.gravity[2])
        return (total_mass * g) / 4.0

    @property
    def max_thrust(self) -> float:
        return float(self._model.actuator_ctrlrange[0, 1])

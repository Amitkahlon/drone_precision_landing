import numpy as np

from .enums import Motor, Sensor

_KP_Z   = 0.5
_KD_Z   = 0.4
_KP_ATT = 2.0
_KD_ATT = 0.5


class _MotorAccessor:
    """Read/write individual motor thrusts via controller.motors.<name>."""

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
    """Read sensor arrays via controller.sensors.<name>."""

    def __init__(self, model, data):
        object.__setattr__(self, '_model', model)
        object.__setattr__(self, '_data', data)

    def _read(self, sensor: Sensor) -> np.ndarray:
        sid = self._model.sensor(sensor).id
        adr = self._model.sensor_adr[sid]
        return self._data.sensordata[adr : adr + self._model.sensor_dim[sid]]

    @property
    def accel(self)  -> np.ndarray: return self._read(Sensor.ACCEL)
    @property
    def gyro(self)   -> np.ndarray: return self._read(Sensor.GYRO)
    @property
    def pos(self)    -> np.ndarray: return self._read(Sensor.POS)
    @property
    def quat(self)   -> np.ndarray: return self._read(Sensor.QUAT)
    @property
    def linvel(self) -> np.ndarray: return self._read(Sensor.LINVEL)
    @property
    def angvel(self) -> np.ndarray: return self._read(Sensor.ANGVEL)


class DroneController:
    def __init__(self, model, data):
        self.model   = model
        self.data    = data
        self.motors  = _MotorAccessor(data.ctrl)
        self.sensors = _SensorAccessor(model, data)
        self._drone_id = model.body("drone").id

    def set_pos_in_world(self, x, y, z):
        self.data.qpos[0] = x
        self.data.qpos[1] = y
        self.data.qpos[2] = z

        # its about its tilt, need to figure it out:
        self.data.qpos[3:7] = [1, 0, 0, 0]

    def compute_hover_thrust(self) -> float:
        total_mass = self.model.body_mass[self._drone_id]
        g = abs(self.model.opt.gravity[2])
        return (total_mass * g) / 4.0

    def hover(self, target_z: float) -> None:
        base  = self.compute_hover_thrust()
        max_T = float(self.model.actuator_ctrlrange[0, 1])

        z  = self.sensors.pos[2]
        vz = self.sensors.linvel[2]
        dZ = _KP_Z * (target_z - z) - _KD_Z * vz

        _, qx, qy, _ = self.sensors.quat
        wx, wy, _    = self.sensors.gyro

        d_roll  = _KP_ATT * qx + _KD_ATT * wx
        d_pitch = _KP_ATT * qy + _KD_ATT * wy

        self.motors.fl = float(np.clip(base + dZ - d_roll + d_pitch, 0.0, max_T))
        self.motors.fr = float(np.clip(base + dZ + d_roll + d_pitch, 0.0, max_T))
        self.motors.br = float(np.clip(base + dZ + d_roll - d_pitch, 0.0, max_T))
        self.motors.bl = float(np.clip(base + dZ - d_roll - d_pitch, 0.0, max_T))

    def print_details(self) -> None:
        print(
            "DroneController | "
            f"FL={self.motors.fl:.3f}  FR={self.motors.fr:.3f}"
            f"  BR={self.motors.br:.3f}  BL={self.motors.bl:.3f}  N"
        )

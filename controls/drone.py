import mujoco
import numpy as np

from .enums import Motor, Sensor


class _MotorAccessor:
    """Controls the four motors. Use drone.motors.<name> to set thrust (0–4 N)."""

    def __init__(self, ctrl):
        object.__setattr__(self, '_ctrl', ctrl)

    @property
    def fl(self) -> float:
        """Front-left motor thrust in Newtons."""
        return float(self._ctrl[Motor.FL])

    @fl.setter
    def fl(self, v: float): self._ctrl[Motor.FL] = v

    @property
    def fr(self) -> float:
        """Front-right motor thrust in Newtons."""
        return float(self._ctrl[Motor.FR])

    @fr.setter
    def fr(self, v: float): self._ctrl[Motor.FR] = v

    @property
    def br(self) -> float:
        """Back-right motor thrust in Newtons."""
        return float(self._ctrl[Motor.BR])

    @br.setter
    def br(self, v: float): self._ctrl[Motor.BR] = v

    @property
    def bl(self) -> float:
        """Back-left motor thrust in Newtons."""
        return float(self._ctrl[Motor.BL])

    @bl.setter
    def bl(self, v: float): self._ctrl[Motor.BL] = v


class _Position:
    """Where the drone is in the world (metres). Read from the pos sensor."""

    def __init__(self, sensors):
        object.__setattr__(self, '_sensors', sensors)

    @property
    def x(self) -> float:
        """World X position in metres."""
        return float(self._sensors.pos[0])

    @property
    def y(self) -> float:
        """World Y position in metres."""
        return float(self._sensors.pos[1])

    @property
    def z(self) -> float:
        """Altitude in metres. 0 = ground level."""
        return float(self._sensors.pos[2])


class _Velocity:
    """How fast the drone is moving in each direction (m/s). Read from the linvel sensor."""

    def __init__(self, sensors):
        object.__setattr__(self, '_sensors', sensors)

    @property
    def forward(self) -> float:
        """Speed along world X axis in m/s. Positive = moving forward."""
        return float(self._sensors.linvel[0])

    @property
    def sideways(self) -> float:
        """Speed along world Y axis in m/s. Positive = moving right."""
        return float(self._sensors.linvel[1])

    @property
    def vertical(self) -> float:
        """Speed along world Z axis in m/s. Positive = rising, negative = falling."""
        return float(self._sensors.linvel[2])


class _Rotation:
    """How fast the drone's angles are changing (rad/s).
    Tilt rates use the gyro sensor (body frame) so they stay correct after yaw.
    Turn rate uses the angvel sensor (world frame Z = body frame Z for any heading)."""

    def __init__(self, sensors):
        object.__setattr__(self, '_sensors', sensors)

    @property
    def sideways_tilt_rate(self) -> float:
        """How fast the drone is tilting left/right in body frame (rad/s). Positive = tilting right."""
        return float(self._sensors.gyro[0])

    @property
    def forward_tilt_rate(self) -> float:
        """How fast the drone is tilting forward/backward in body frame (rad/s). Positive = tilting forward."""
        return float(self._sensors.gyro[1])

    @property
    def turn_rate(self) -> float:
        """How fast the drone is spinning in place (rad/s). Positive = turning right."""
        return float(self._sensors.angvel[2])


class _Orientation:
    """The drone's current tilt and heading angles (radians). Read from the quat sensor."""

    def __init__(self, sensors):
        object.__setattr__(self, '_sensors', sensors)

    @property
    def lean_sideways(self) -> float:
        """How much the drone is tilted left/right (radians). Positive = leaning right."""
        return float(self._sensors.quat[1])

    @property
    def lean_forward(self) -> float:
        """How much the drone is tilted forward/backward (radians). Positive = leaning forward."""
        return float(self._sensors.quat[2])

    @property
    def heading(self) -> float:
        """Direction the drone is facing (radians). 0 = forward, positive = turning right."""
        w, x, y, z = self._sensors.quat
        return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


class _SensorAccessor:
    """Internal raw sensor reader. Not exposed on Drone — use position/velocity/rotation/orientation instead."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        object.__setattr__(self, '_model', model)
        object.__setattr__(self, '_data', data)

    def _read(self, sensor: Sensor) -> np.ndarray:
        sid = self._model.sensor(sensor).id
        adr = self._model.sensor_adr[sid]
        return self._data.sensordata[adr: adr + self._model.sensor_dim[sid]]

    @property
    def pos(self) -> np.ndarray: return self._read(Sensor.POS)

    @property
    def quat(self) -> np.ndarray: return self._read(Sensor.QUAT)

    @property
    def linvel(self) -> np.ndarray: return self._read(Sensor.LINVEL)

    @property
    def angvel(self) -> np.ndarray: return self._read(Sensor.ANGVEL)

    @property
    def gyro(self) -> np.ndarray: return self._read(Sensor.GYRO)


class Drone:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self._model = model
        self._data = data
        self._body_id = model.body("drone").id
        joint_id = model.body_jntadr[self._body_id]
        self.qpos_adr = int(model.jnt_qposadr[joint_id])
        self.motors = _MotorAccessor(data.ctrl)

        _sensors = _SensorAccessor(model, data)
        self.position = _Position(_sensors)
        self.velocity = _Velocity(_sensors)
        self.rotation = _Rotation(_sensors)
        self.orientation = _Orientation(_sensors)

    def compute_hover_thrust(self) -> float:
        """The thrust each motor needs to produce just to hold the drone in the air."""
        total_mass = self._model.body_mass[self._body_id]
        g = abs(self._model.opt.gravity[2])
        return (total_mass * g) / 4.0

    @property
    def max_thrust(self) -> float:
        """Maximum thrust a single motor can produce in Newtons."""
        return float(self._model.actuator_ctrlrange[0, 1])

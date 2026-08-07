import mujoco
import numpy as np

from ..enums import Sensor


class SensorArray:
    """Named access to the IMU sensor readings declared in the MJCF model.

    Each property returns a live view into `data.sensordata`, sliced by the
    address and width MuJoCo assigned to that sensor.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self._model = model
        self._data = data

    def _read(self, sensor: Sensor) -> np.ndarray:
        """Read a sensor from the model."""
        sensor_id = self._model.sensor(sensor).id
        address = self._model.sensor_adr[sensor_id]
        return self._data.sensordata[address:address + self._model.sensor_dim[sensor_id]]

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

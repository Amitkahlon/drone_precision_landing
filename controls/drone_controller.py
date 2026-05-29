import numpy as np

from .drone import Drone
from .enums import DroneState

_KP_Z      = 0.5
_KD_Z      = 0.4
_KP_ATT    = 2.0
_KD_ATT    = 0.5
_KP_YAW    = 2.0
_KD_YAW    = 0.5
_KP_XY     = 0.15
_KD_XY     = 0.3
_MAX_D_YAW = 0.5
_MAX_TILT  = 0.15

_ARRIVAL_THRESHOLD = 0.05
_LANDING_THRESHOLD = 0.12   # chassis half-height is 0.055, so resting z ≈ 0.055
_LANDING_SINK      = 0.5    # target this far below current z → ~0.5 m/s descent


class DroneController:
    def __init__(self, drone: Drone):
        self.drone          = drone
        self.state          = DroneState.GROUNDED
        self._target_z:     float = 0.0
        self._target_x:     float = 0.0
        self._target_y:     float = 0.0
        self._target_roll:  float = 0.0
        self._target_pitch: float = 0.0

    # --- public commands ---

    def take_off(self, target_z: float) -> None:
        self._target_z = target_z
        self.state     = DroneState.TAKING_OFF

    def land(self) -> None:
        self._target_z = 0.0
        self.state     = DroneState.LANDING

    def hover(self) -> None:
        self._target_z = self.drone.get_z()
        self._target_x = self.drone.get_x()
        self._target_y = self.drone.get_y()
        self.state     = DroneState.HOVERING

    def fly(self, angle_deg: float, magnitude: float) -> None:
        """Fly in any direction by tilting. 0=forward, 90=right, 180=backward, 270=left."""
        magnitude          = float(np.clip(magnitude, 0.0, 1.0))
        angle_rad          = np.deg2rad(angle_deg)
        self._target_pitch = _MAX_TILT * magnitude * np.cos(angle_rad)
        self._target_roll  = _MAX_TILT * magnitude * np.sin(angle_rad)
        self._apply_motors()

    def rotate(self, yaw_rate: float) -> None:
        self._target_roll  = 0.0
        self._target_pitch = 0.0
        actual = self.drone.sensors.angvel[2]
        d_yaw  = float(np.clip(_KP_YAW * (yaw_rate - actual), -_MAX_D_YAW, _MAX_D_YAW))
        self._apply_motors(d_yaw)

    def rotate_to(self, target_yaw: float) -> None:
        self._target_roll  = 0.0
        self._target_pitch = 0.0
        yaw_error = (target_yaw - self._get_yaw() + np.pi) % (2 * np.pi) - np.pi
        yaw_rate  = self.drone.sensors.angvel[2]
        d_yaw     = float(np.clip(_KP_YAW * yaw_error - _KD_YAW * yaw_rate, -_MAX_D_YAW, _MAX_D_YAW))
        self._apply_motors(d_yaw)

    def step(self) -> None:
        if self.state == DroneState.GROUNDED:
            self._apply_grounded()
        elif self.state == DroneState.TAKING_OFF:
            self._apply_taking_off()
        elif self.state == DroneState.HOVERING:
            self._apply_hovering()
        elif self.state == DroneState.LANDING:
            self._apply_landing()

    # --- state behaviours ---

    def _apply_grounded(self) -> None:
        self.drone.motors.fl = 0.0
        self.drone.motors.fr = 0.0
        self.drone.motors.br = 0.0
        self.drone.motors.bl = 0.0

    def _apply_taking_off(self) -> None:
        self._apply_motors()
        if abs(self.drone.get_z() - self._target_z) < _ARRIVAL_THRESHOLD:
            self._target_x = self.drone.get_x()
            self._target_y = self.drone.get_y()
            self.state     = DroneState.HOVERING

    def _apply_hovering(self) -> None:
        self._update_position_hold()
        self._apply_motors()

    def _apply_landing(self) -> None:
        self._target_z = max(0.0, self.drone.get_z() - _LANDING_SINK)
        self._update_position_hold()
        self._apply_motors()
        if self.drone.get_z() < _LANDING_THRESHOLD:
            self.state = DroneState.GROUNDED
            self._apply_grounded()

    # --- internals ---

    def _update_position_hold(self) -> None:
        error_x = self._target_x - self.drone.get_x()
        error_y = self._target_y - self.drone.get_y()
        vx = self.drone.sensors.linvel[0]
        vy = self.drone.sensors.linvel[1]
        self._target_pitch = float(np.clip( _KP_XY * error_x - _KD_XY * vx, -_MAX_TILT, _MAX_TILT))
        self._target_roll  = float(np.clip(-(_KP_XY * error_y - _KD_XY * vy), -_MAX_TILT, _MAX_TILT))

    def _get_yaw(self) -> float:
        w, x, y, z = self.drone.sensors.quat
        return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

    def _apply_motors(self, d_yaw: float = 0.0) -> None:
        base  = self.drone.compute_hover_thrust()
        max_T = self.drone.max_thrust

        z  = self.drone.sensors.pos[2]
        vz = self.drone.sensors.linvel[2]
        dZ = _KP_Z * (self._target_z - z) - _KD_Z * vz

        _, qx, qy, _ = self.drone.sensors.quat
        wx, wy, _    = self.drone.sensors.gyro

        d_roll  = _KP_ATT * (qx - self._target_roll)  + _KD_ATT * wx
        d_pitch = _KP_ATT * (qy - self._target_pitch) + _KD_ATT * wy

        self.drone.motors.fl = float(np.clip(base + dZ - d_roll + d_pitch + d_yaw, 0.0, max_T))
        self.drone.motors.fr = float(np.clip(base + dZ + d_roll + d_pitch - d_yaw, 0.0, max_T))
        self.drone.motors.br = float(np.clip(base + dZ + d_roll - d_pitch + d_yaw, 0.0, max_T))
        self.drone.motors.bl = float(np.clip(base + dZ - d_roll - d_pitch - d_yaw, 0.0, max_T))

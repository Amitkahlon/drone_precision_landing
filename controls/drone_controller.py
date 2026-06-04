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
_LANDING_THRESHOLD = 0.12
_LANDING_SINK      = 0.5


class DroneController:
    def __init__(self, drone: Drone):
        self.drone            = drone
        self.state            = DroneState.GROUNDED
        self._target_z:       float = 0.0
        self._target_x:       float = 0.0
        self._target_y:       float = 0.0
        self._target_roll:    float = 0.0
        self._target_pitch:   float = 0.0
        self._yaw_correction: float = 0.0

    # --- public commands ---

    def take_off(self, target_z: float) -> None:
        self._target_z       = target_z
        self._yaw_correction = 0.0
        self.state           = DroneState.TAKING_OFF

    def land(self) -> None:
        self._target_z       = 0.0
        self._yaw_correction = 0.0
        self.state           = DroneState.LANDING

    def hover(self) -> None:
        self._target_z       = self.drone.position.z
        self._target_x       = self.drone.position.x
        self._target_y       = self.drone.position.y
        self._yaw_correction = 0.0
        self.state           = DroneState.HOVERING

    def fly(self, angle_deg: float, magnitude: float) -> None:
        """Fly in any direction by tilting. 0=forward, 90=right, 180=backward, 270=left."""
        magnitude          = float(np.clip(magnitude, 0.0, 1.0))
        angle_rad          = np.deg2rad(angle_deg)
        self._target_pitch = float(_MAX_TILT * magnitude * np.cos(angle_rad))
        self._target_roll  = float(_MAX_TILT * magnitude * np.sin(angle_rad))
        self._update_motors()

    def rotate(self, yaw_rate: float) -> None:
        self._target_roll    = 0.0
        self._target_pitch   = 0.0
        self._yaw_correction = float(np.clip(
            _KP_YAW * (yaw_rate - self.drone.rotation.turn_rate),
            -_MAX_D_YAW, _MAX_D_YAW
        ))
        self._update_motors()

    def rotate_to(self, target_yaw: float) -> None:
        self._target_roll  = 0.0
        self._target_pitch = 0.0
        yaw_error          = (target_yaw - self.drone.orientation.heading + np.pi) % (2 * np.pi) - np.pi
        self._yaw_correction = float(np.clip(
            _KP_YAW * yaw_error - _KD_YAW * self.drone.rotation.turn_rate,
            -_MAX_D_YAW, _MAX_D_YAW
        ))
        self._update_motors()

    def step(self) -> None:
        if self.state == DroneState.GROUNDED:
            self._motors_off()
        elif self.state == DroneState.TAKING_OFF:
            self._rise_to_target()
        elif self.state == DroneState.HOVERING:
            self._hold_position()
        elif self.state == DroneState.LANDING:
            self._descend_to_ground()

    # --- state behaviours ---

    def _motors_off(self) -> None:
        self.drone.motors.fl = 0.0
        self.drone.motors.fr = 0.0
        self.drone.motors.br = 0.0
        self.drone.motors.bl = 0.0

    def _rise_to_target(self) -> None:
        self._update_motors()
        if abs(self.drone.position.z - self._target_z) < _ARRIVAL_THRESHOLD:
            self._target_x = self.drone.position.x
            self._target_y = self.drone.position.y
            self.state     = DroneState.HOVERING

    def _hold_position(self) -> None:
        self._yaw_correction = float(np.clip(
            -_KP_YAW * self.drone.rotation.turn_rate,
            -_MAX_D_YAW, _MAX_D_YAW
        ))
        self._compute_tilt_from_position()
        self._update_motors()

    def _descend_to_ground(self) -> None:
        self._target_z = max(0.0, self.drone.position.z - _LANDING_SINK)
        self._compute_tilt_from_position()
        self._update_motors()
        if self.drone.position.z < _LANDING_THRESHOLD:
            self.state = DroneState.GROUNDED
            self._motors_off()

    # --- motor control ---

    def _compute_thrust(self) -> float:
        altitude_error = self._target_z - self.drone.position.z
        return self.drone.compute_hover_thrust() + _KP_Z * altitude_error - _KD_Z * self.drone.velocity.vertical

    def _compute_roll(self) -> float:
        tilt_error = self.drone.orientation.lean_sideways - self._target_roll
        return _KP_ATT * tilt_error + _KD_ATT * self.drone.rotation.sideways_tilt_rate

    def _compute_pitch(self) -> float:
        tilt_error = self.drone.orientation.lean_forward - self._target_pitch
        return _KP_ATT * tilt_error + _KD_ATT * self.drone.rotation.forward_tilt_rate

    def _mix(self, thrust: float, roll: float, pitch: float, yaw: float) -> None:
        max_thrust = self.drone.max_thrust
        self.drone.motors.fl = float(np.clip(thrust - roll + pitch + yaw, 0.0, max_thrust))
        self.drone.motors.fr = float(np.clip(thrust + roll + pitch - yaw, 0.0, max_thrust))
        self.drone.motors.bl = float(np.clip(thrust - roll - pitch - yaw, 0.0, max_thrust))
        self.drone.motors.br = float(np.clip(thrust + roll - pitch + yaw, 0.0, max_thrust))

    def _update_motors(self) -> None:
        thrust = self._compute_thrust()
        roll   = self._compute_roll()
        pitch  = self._compute_pitch()
        self._mix(thrust, roll, pitch, self._yaw_correction)

    # --- helpers ---

    def _compute_tilt_from_position(self) -> None:
        error_x = self._target_x - self.drone.position.x
        error_y = self._target_y - self.drone.position.y
        self._target_pitch = float(np.clip( _KP_XY * error_x - _KD_XY * self.drone.velocity.forward,   -_MAX_TILT, _MAX_TILT))
        self._target_roll  = float(np.clip(-(_KP_XY * error_y - _KD_XY * self.drone.velocity.sideways), -_MAX_TILT, _MAX_TILT))

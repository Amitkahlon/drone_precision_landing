import numpy as np

from ..drone import Drone
from .gains import (
    KD_ATTITUDE,
    KD_XY,
    KD_Z,
    KP_ATTITUDE,
    KP_XY,
    KP_YAW,
    KP_Z,
    MAX_TILT,
    MAX_YAW_THRUST_SPLIT,
)


class ThrustMixer:
    """Converts flight setpoints into the four individual motor thrusts.

    Holds every PD loop that produces a number the motors act on, leaving the
    controller responsible only for deciding which setpoints apply right now.
    """

    def __init__(self, drone: Drone):
        self._drone = drone

    def cut_thrust(self) -> None:
        self._drone.motors.set_all(0.0, 0.0, 0.0, 0.0)

    def apply(
            self,
            target_z: float,
            target_roll: float,
            target_pitch: float,
            yaw_thrust_split: float = 0.0,
    ) -> None:
        """Mix altitude, attitude and yaw corrections onto the rotors.

        Diagonal rotor pairs spin opposite ways, so splitting thrust across the
        diagonals yaws the airframe via the rotors' reactive torque.
        """
        hover = self._drone.compute_hover_thrust()
        ceiling = self._drone.max_thrust

        climb = self._vertical_correction(target_z)
        roll, pitch = self._attitude_correction(target_roll, target_pitch)

        self._drone.motors.set_all(
            fl=float(np.clip(hover + climb - roll + pitch + yaw_thrust_split, 0.0, ceiling)),
            fr=float(np.clip(hover + climb + roll + pitch - yaw_thrust_split, 0.0, ceiling)),
            br=float(np.clip(hover + climb + roll - pitch + yaw_thrust_split, 0.0, ceiling)),
            bl=float(np.clip(hover + climb - roll - pitch - yaw_thrust_split, 0.0, ceiling)),
        )

    def tilt_for_position_hold(
            self,
            target_x: float,
            target_y: float,
            target_velocity: np.ndarray,
    ) -> tuple[float, float]:
        """Return the (roll, pitch) setpoint that flies the drone to a point.

        `target_velocity` is fed forward and subtracted from the drone's own
        velocity, so chasing a moving platform settles at the platform's speed
        instead of fighting to stand still over a target that keeps sliding away.
        """
        error_x = target_x - self._drone.get_x()
        error_y = target_y - self._drone.get_y()
        velocity_x = self._drone.sensors.linvel[0]
        velocity_y = self._drone.sensors.linvel[1]

        pitch = KP_XY * error_x - KD_XY * (velocity_x - target_velocity[0])
        roll = -(KP_XY * error_y - KD_XY * (velocity_y - target_velocity[1]))
        return (
            float(np.clip(roll, -MAX_TILT, MAX_TILT)),
            float(np.clip(pitch, -MAX_TILT, MAX_TILT)),
        )

    def yaw_thrust_split_for_rate(self, target_yaw_rate: float) -> float:
        yaw_rate = self._drone.sensors.angvel[2]
        return float(np.clip(
            KP_YAW * (target_yaw_rate - yaw_rate),
            -MAX_YAW_THRUST_SPLIT,
            MAX_YAW_THRUST_SPLIT,
        ))

    def _vertical_correction(self, target_z: float) -> float:
        altitude = self._drone.sensors.pos[2]
        climb_rate = self._drone.sensors.linvel[2]
        return KP_Z * (target_z - altitude) - KD_Z * climb_rate

    def _attitude_correction(
            self,
            target_roll: float,
            target_pitch: float,
    ) -> tuple[float, float]:
        """Roll and pitch thrust corrections from the quaternion tilt terms.

        For near-level flight the quaternion's x and y components are directly
        proportional to roll and pitch, so they are used as the error directly
        rather than converting to Euler angles.
        """
        _, tilt_x, tilt_y, _ = self._drone.sensors.quat
        rate_x, rate_y, _ = self._drone.sensors.gyro
        return (
            KP_ATTITUDE * (tilt_x - target_roll) + KD_ATTITUDE * rate_x,
            KP_ATTITUDE * (tilt_y - target_pitch) + KD_ATTITUDE * rate_y,
        )

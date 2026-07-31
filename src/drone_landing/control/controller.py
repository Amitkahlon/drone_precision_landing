from typing import Protocol

import numpy as np

from ..drone import Drone
from ..enums import DroneState
from .gains import (
    ALIGNMENT_THRESHOLD_M,
    ARRIVAL_THRESHOLD_M,
    LANDING_SINK_M,
    MAX_TILT,
    TOUCHDOWN_THRESHOLD_M,
    TRACKING_OFFSET_M,
)
from .mixer import ThrustMixer


class LandingTarget(Protocol):
    """Anything the drone can chase and land on, such as a moving platform."""

    @property
    def position(self) -> np.ndarray: ...

    @property
    def velocity(self) -> np.ndarray: ...


class DroneController:
    """Flight state machine.

    Owns which setpoints are active in each state and when to change state;
    the arithmetic that turns those setpoints into thrust lives in ThrustMixer.
    The nominal sequence for a landing run is

        GROUNDED -> TAKING_OFF -> TRACKING -> LANDING -> GROUNDED

    with HOVERING substituting for TRACKING when no target has been set.
    """

    def __init__(self, drone: Drone):
        self.drone = drone
        self.state = DroneState.GROUNDED
        self._mixer = ThrustMixer(drone)
        self._target_z: float = 0.0
        self._target_x: float = 0.0
        self._target_y: float = 0.0
        self._target_roll: float = 0.0
        self._target_pitch: float = 0.0
        self._landing_target: LandingTarget | None = None

    # --- commands ---

    def set_target(self, target: LandingTarget) -> None:
        self._landing_target = target

    def take_off(self, target_z: float) -> None:
        self._target_z = target_z
        self.state = DroneState.TAKING_OFF

    def land(self) -> None:
        self._target_z = 0.0
        self.state = DroneState.LANDING

    def hover(self) -> None:
        self._target_z = self.drone.get_z()
        self._target_x = self.drone.get_x()
        self._target_y = self.drone.get_y()
        self.state = DroneState.HOVERING

    def track(self) -> None:
        self.state = DroneState.TRACKING

    def fly(self, angle_deg: float, magnitude: float) -> None:
        """Fly in any direction by tilting. 0=forward, 90=right, 180=backward, 270=left."""
        magnitude = float(np.clip(magnitude, 0.0, 1.0))
        angle_rad = np.deg2rad(angle_deg)
        self._target_pitch = MAX_TILT * magnitude * np.cos(angle_rad)
        self._target_roll = MAX_TILT * magnitude * np.sin(angle_rad)
        self._apply_motors()

    def rotate(self, yaw_rate: float) -> None:
        self._target_roll = 0.0
        self._target_pitch = 0.0
        self._apply_motors(self._mixer.yaw_thrust_split_for_rate(yaw_rate))

    # --- per-step update ---

    def step(self) -> None:
        handler = {
            DroneState.GROUNDED: self._hold_grounded,
            DroneState.TAKING_OFF: self._climb_to_altitude,
            DroneState.HOVERING: self._hold_position,
            DroneState.TRACKING: self._chase_target,
            DroneState.LANDING: self._descend,
        }.get(self.state)
        if handler is not None:
            handler()

    # --- state behaviours ---

    def _hold_grounded(self) -> None:
        self._mixer.cut_thrust()

    def _climb_to_altitude(self) -> None:
        self._apply_motors()
        if abs(self.drone.get_z() - self._target_z) < ARRIVAL_THRESHOLD_M:
            self._target_x = self.drone.get_x()
            self._target_y = self.drone.get_y()
            self.state = DroneState.TRACKING if self._landing_target else DroneState.HOVERING

    def _hold_position(self) -> None:
        self._update_position_hold()
        self._apply_motors()

    def _chase_target(self) -> None:
        target = self._landing_target
        if target is None:
            self._hold_position()
            return

        position = target.position
        self._target_x = float(position[0])
        self._target_y = float(position[1])
        self._target_z = float(position[2]) + TRACKING_OFFSET_M

        self._update_position_hold()
        self._apply_motors()

        if self._horizontal_error() < ALIGNMENT_THRESHOLD_M:
            self.state = DroneState.LANDING

    def _descend(self) -> None:
        platform_z = 0.0
        if self._landing_target is not None:
            position = self._landing_target.position
            self._target_x = float(position[0])
            self._target_y = float(position[1])
            platform_z = float(position[2])

        # Chase a point a fixed distance below the drone rather than the floor
        # itself, which turns the altitude hold into a bounded sink rate.
        self._target_z = max(platform_z, self.drone.get_z() - LANDING_SINK_M)
        self._update_position_hold()
        self._apply_motors()

        if self.drone.get_z() < platform_z + TOUCHDOWN_THRESHOLD_M:
            self.state = DroneState.GROUNDED
            self._hold_grounded()

    # --- internals ---

    def _update_position_hold(self) -> None:
        target_velocity = (
            self._landing_target.velocity if self._landing_target else np.zeros(3)
        )
        self._target_roll, self._target_pitch = self._mixer.tilt_for_position_hold(
            self._target_x, self._target_y, target_velocity
        )

    def _horizontal_error(self) -> float:
        return float(np.sqrt(
            (self.drone.get_x() - self._target_x) ** 2 +
            (self.drone.get_y() - self._target_y) ** 2
        ))

    def _apply_motors(self, yaw_thrust_split: float = 0.0) -> None:
        self._mixer.apply(
            self._target_z,
            self._target_roll,
            self._target_pitch,
            yaw_thrust_split,
        )

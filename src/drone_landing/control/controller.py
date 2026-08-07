from typing import Protocol

import numpy as np

from ..drone import Drone
from ..enums import DroneState
from .gains import (
    ALIGNMENT_THRESHOLD_M,
    ARRIVAL_THRESHOLD_M,
    LANDING_SINK_M,
    LAUNCH_CLEARANCE_M,
    LAUNCH_WINDOW_S,
    MAX_REALIGN_ATTEMPTS,
    MAX_TILT,
    REALIGN_CLEARANCE_M,
    REALIGN_SETTLE_S,
    REALIGN_THRESHOLD_M,
    REALIGN_TIMEOUT_S,
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

        GROUNDED -> WAITING_TO_LAUNCH -> TAKING_OFF -> TRACKING -> LANDING -> GROUNDED

    with HOVERING substituting for TRACKING when no target has been set,
    REALIGNING taken from LANDING and back whenever the platform pulls far
    enough out from under a descent to be worth breaking off, and
    WAITING_TO_LAUNCH passed straight through when the pad is already clear.
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
        self._realign_attempts: int = 0
        self._aligned_steps: int = 0
        self._realign_steps: int = 0
        self._realign_z: float = 0.0
        self._gave_up: bool = False

    @property
    def gave_up(self) -> bool:
        """True once too many descents have been abandoned to keep trying."""
        return self._gave_up

    # --- commands ---

    def set_target(self, target: LandingTarget) -> None:
        self._landing_target = target

    def take_off(self, target_z: float) -> None:
        self._target_z = target_z
        self._target_x = self.drone.get_x()
        self._target_y = self.drone.get_y()
        self._realign_attempts = 0
        self._aligned_steps = 0
        self._realign_steps = 0
        self._gave_up = False
        self.state = DroneState.WAITING_TO_LAUNCH

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
            DroneState.REALIGNING: self._realign,
            DroneState.WAITING_TO_LAUNCH: self._wait_for_clear_pad,
        }.get(self.state)
        if handler is not None:
            handler()

    # --- state behaviours ---

    def _hold_grounded(self) -> None:
        self._mixer.cut_thrust()

    def _wait_for_clear_pad(self) -> None:
        """Stay on the ground while the platform is passing over the launch point.

        Platforms are mocap bodies, so one crossing the pad mid-climb is
        unaffected by the collision and simply drags the drone along with it.
        """
        self._mixer.cut_thrust()
        if self._pad_is_clear():
            self.state = DroneState.TAKING_OFF
            self._climb_to_altitude()

    def _climb_to_altitude(self) -> None:
        # Zero feedforward rather than _update_position_hold: the climb holds the
        # launch point, and feeding the platform's velocity into a stationary
        # setpoint would tilt the drone after a target it is not chasing yet.
        self._target_roll, self._target_pitch = self._mixer.tilt_for_position_hold(
            self._target_x, self._target_y, np.zeros(3)
        )
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

        if self._horizontal_error() > REALIGN_THRESHOLD_M:
            self._abandon_descent()
            return

        # Chase a point a fixed distance below the drone rather than the floor
        # itself, which turns the altitude hold into a bounded sink rate.
        self._target_z = max(platform_z, self.drone.get_z() - LANDING_SINK_M)
        self._update_position_hold()
        self._apply_motors()

        if self.drone.get_z() < platform_z + TOUCHDOWN_THRESHOLD_M:
            self.state = DroneState.GROUNDED
            self._hold_grounded()

    def _realign(self) -> None:
        """Hold altitude over the platform until the descent can be resumed.

        Holding rather than climbing back to TRACKING_OFFSET_M keeps whatever
        altitude the abandoned descent already bought, which is the difference
        between re-attempting within the platform's current leg and arriving
        after it has turned again.
        """
        if self._landing_target is not None:
            position = self._landing_target.position
            self._target_x = float(position[0])
            self._target_y = float(position[1])

        # The altitude latched on entry, not the current one: chasing the
        # current altitude leaves the vertical loop with no reference to
        # restore, and the thrust a hard lateral correction tilts away is then
        # never recovered, so the drone sinks into the platform mid-realign.
        self._target_z = self._realign_z
        self._update_position_hold()
        self._apply_motors()

        if self._gave_up:
            return

        self._realign_steps += 1
        if self._realign_steps > self._seconds_in_steps(REALIGN_TIMEOUT_S):
            self._gave_up = True
            return

        if self._horizontal_error() < ALIGNMENT_THRESHOLD_M:
            self._aligned_steps += 1
        else:
            self._aligned_steps = 0

        if self._aligned_steps >= self._required_settle_steps():
            self.state = DroneState.LANDING

    # --- internals ---

    def _pad_is_clear(self) -> bool:
        """True when the target stays clear of the launch point for the climb.

        Legs are straight and the platform does not accelerate, so extrapolating
        from its current velocity is exact within a leg, and the closest approach
        over the window has a closed form.
        """
        if self._landing_target is None:
            return True

        offset = self._landing_target.position[:2] - np.array(
            [self._target_x, self._target_y]
        )
        velocity = self._landing_target.velocity[:2]
        speed_squared = float(velocity @ velocity)

        closest = 0.0
        if speed_squared > 1e-12:
            closest = float(np.clip(
                -(offset @ velocity) / speed_squared, 0.0, LAUNCH_WINDOW_S
            ))
        return float(np.linalg.norm(offset + velocity * closest)) > LAUNCH_CLEARANCE_M

    def _abandon_descent(self) -> None:
        """Give up on this descent and start holding for another attempt."""
        platform_z = 0.0
        if self._landing_target is not None:
            platform_z = float(self._landing_target.position[2])

        self._realign_attempts += 1
        self._aligned_steps = 0
        self._realign_steps = 0
        self._realign_z = max(self.drone.get_z(), platform_z + REALIGN_CLEARANCE_M)
        self._gave_up = self._realign_attempts > MAX_REALIGN_ATTEMPTS
        self.state = DroneState.REALIGNING
        self._realign()

    def _required_settle_steps(self) -> int:
        return self._seconds_in_steps(REALIGN_SETTLE_S * self._realign_attempts)

    def _seconds_in_steps(self, seconds: float) -> int:
        return int(seconds / self.drone.timestep)

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

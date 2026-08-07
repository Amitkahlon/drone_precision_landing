from enum import IntEnum, StrEnum, auto, Enum


class Motor(IntEnum):
    """Drone motors."""
    FL = 0  # front-left  (green,  CCW)
    FR = 1  # front-right (blue,   CW)
    BR = 2  # back-right  (yellow, CCW)
    BL = 3  # back-left   (orange, CW)


class DroneState(Enum):
    """Drone state machine."""
    GROUNDED = auto()
    TAKING_OFF = auto()
    HOVERING = auto()
    TRACKING = auto()
    LANDING = auto()
    REALIGNING = auto()
    WAITING_TO_LAUNCH = auto()


class Sensor(StrEnum):
    """Drone sensors."""
    ACCEL = "accel"  # accelerometer
    GYRO = "gyro"  # gyroscope
    POS = "pos"  # position
    QUAT = "quat"  # quaternion
    LINVEL = "linvel"  # linear velocity
    ANGVEL = "angvel"  # angular velocity

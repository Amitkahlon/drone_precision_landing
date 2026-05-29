from enum import IntEnum, StrEnum, auto, Enum


class Motor(IntEnum):
    FL = 0  # front-left  (green,  CCW)
    FR = 1  # front-right (blue,   CW)
    BR = 2  # back-right  (yellow, CCW)
    BL = 3  # back-left   (orange, CW)


class DroneState(Enum):
    GROUNDED = auto()
    TAKING_OFF = auto()
    HOVERING = auto()
    FLYING = auto()
    TRACKING = auto()
    LANDING = auto()


class Sensor(StrEnum):
    ACCEL = "accel"
    GYRO = "gyro"
    POS = "pos"
    QUAT = "quat"
    LINVEL = "linvel"
    ANGVEL = "angvel"

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
    LANDING = auto()


class Sensor(StrEnum):
    POS    = "pos"    # where am I?          (3,) m
    QUAT   = "quat"   # which way am I tilted? (4,) quaternion
    LINVEL = "linvel" # how fast am I moving? (3,) m/s
    ANGVEL = "angvel" # how fast am I rotating in world frame? (3,) rad/s
    GYRO   = "gyro"   # how fast am I rotating in body frame? (3,) rad/s

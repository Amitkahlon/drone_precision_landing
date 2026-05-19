from enum import IntEnum, StrEnum


class Motor(IntEnum):
    FL = 0  # front-left  (green,  CCW)
    FR = 1  # front-right (blue,   CW)
    BR = 2  # back-right  (yellow, CCW)
    BL = 3  # back-left   (orange, CW)


class Sensor(StrEnum):
    ACCEL  = "accel"   # accelerometer  (3,) m/s²
    GYRO   = "gyro"    # gyroscope      (3,) rad/s
    POS    = "pos"     # position       (3,) m
    QUAT   = "quat"    # orientation    (4,) quaternion
    LINVEL = "linvel"  # linear velocity  (3,) m/s
    ANGVEL = "angvel"  # angular velocity (3,) rad/s

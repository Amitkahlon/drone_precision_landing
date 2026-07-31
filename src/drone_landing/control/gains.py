"""Tuning constants for the landing controller.

The four cascaded loops are, from innermost out: attitude (quaternion tilt to
differential thrust), vertical position, yaw rate, and horizontal position
(which commands tilt rather than thrust). Changing any of these changes flight
behaviour, so they are kept together rather than inlined at their use sites.
"""

# Vertical position hold: altitude error and climb rate to collective thrust.
KP_Z = 0.5
KD_Z = 0.4

# Attitude hold: quaternion tilt and body rate to differential thrust.
KP_ATTITUDE = 2.0
KD_ATTITUDE = 0.5

# Yaw rate hold.
KP_YAW = 2.0

# Horizontal position hold, which outputs a tilt setpoint rather than thrust.
KP_XY = 0.15
KD_XY = 0.3

# Authority limits.
MAX_YAW_THRUST_SPLIT = 0.5
MAX_TILT = 0.15

# Altitude error (m) below which a climb counts as having arrived.
ARRIVAL_THRESHOLD_M = 0.05

# Height (m) above the platform at which touchdown is declared. The chassis
# half-height is 0.055, so a drone at rest on a platform sits at about 0.055.
TOUCHDOWN_THRESHOLD_M = 0.12

# Distance (m) below the current altitude to aim at while descending, which
# works out to roughly a 0.5 m/s sink rate.
LANDING_SINK_M = 0.5

# Height (m) above the platform to hold while still chasing it laterally.
TRACKING_OFFSET_M = 1.5

# Horizontal error (m) below which the descent is allowed to start.
ALIGNMENT_THRESHOLD_M = 0.3

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

# Horizontal error (m) above which a descent in progress is abandoned. Kept
# above ALIGNMENT_THRESHOLD_M so abandoning and resuming are hysteretic rather
# than flipping every step around one threshold.
REALIGN_THRESHOLD_M = 0.5

# Time (s) the drone must hold alignment before a descent resumes, multiplied by
# the number of attempts so far. The platform's route is cyclic, so a retry that
# always took the same time would keep re-approaching at the same point in that
# cycle and be cut off by the same turn; growing the wait shifts the phase.
REALIGN_SETTLE_S = 0.2

# Least height (m) above the platform to hold while lining up again. A descent
# abandoned just short of touchdown would otherwise be retried from a height
# with no room to tilt, and the lateral correction would drag the airframe
# across the platform instead of over it.
REALIGN_CLEARANCE_M = 0.8

# Descents that may be abandoned before the controller stops trying.
MAX_REALIGN_ATTEMPTS = 4

# Time (s) one attempt may spend trying to line up again. A drone that has lost
# the platform holds position rather than descending, so without this bound it
# would sit there until the mission timeout instead of reporting that it is
# beaten. Deliberately well above the time a recoverable attempt takes.
REALIGN_TIMEOUT_S = 15.0

# Distance (m) the platform must keep from the launch point before a climb may
# start. Platform routes are not generated around the drone's start, so one can
# sweep straight over the pad; a mocap body is unaffected by the collision and
# simply bulldozes the drone. A 1x1 platform has a 0.707 m half-diagonal and the
# rotors reach past the chassis, so this is that plus margin.
LAUNCH_CLEARANCE_M = 1.2

# How far ahead (s) the pad must stay clear for. A clean climb to 3 m takes
# about 1.8 s, so this covers it with room to spare.
LAUNCH_WINDOW_S = 2.5

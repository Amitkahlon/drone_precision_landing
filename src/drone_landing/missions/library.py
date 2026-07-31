"""Hand-built missions, kept as reference scenarios alongside the random generator.

Each is a fixed platform route, so they are the scenarios to reach for when
comparing controller changes against a known trajectory.
"""

from collections.abc import Callable

from ..scene import Mission, Platform

_ALTITUDE_M = 0.5
_STRAIGHT_LINE_SPEED = 0.8
_SQUARE_SPEED = 1.2


def straight_line() -> Mission:
    """Platform shuttling back and forth along a single line."""
    mission = Mission(
        start=(0, 0, 0),
        platform=Platform(width=1.0, depth=1.0, thickness=0.05, color=(0.9, 0.1, 0.1, 1.0)),
    )
    mission.add_checkpoint((5, 5, _ALTITUDE_M), speed=_STRAIGHT_LINE_SPEED)
    mission.add_checkpoint((-5, 5, _ALTITUDE_M), speed=_STRAIGHT_LINE_SPEED)
    return mission


def square_circuit() -> Mission:
    """Platform running a closed square, faster than the straight-line route."""
    mission = Mission(
        start=(0, 0, 0),
        platform=Platform(width=1.0, depth=1.0, thickness=0.05, color=(0.1, 0.5, 0.9, 1.0)),
    )
    for x, y in ((4, 4), (-4, 4), (-4, -4), (4, -4)):
        mission.add_checkpoint((x, y, _ALTITUDE_M), speed=_SQUARE_SPEED)
    return mission


# mission1/mission2 are kept as aliases for the names these routes had when they
# were standalone scripts.
MISSIONS: dict[str, Callable[[], Mission]] = {
    "straight-line": straight_line,
    "mission1": straight_line,
    "square": square_circuit,
    "mission2": square_circuit,
}


def build_mission(name: str) -> Mission:
    if name not in MISSIONS:
        raise KeyError(f"unknown mission {name!r}; choose from {', '.join(sorted(MISSIONS))}")
    return MISSIONS[name]()

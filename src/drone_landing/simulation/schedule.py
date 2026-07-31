"""A repeating schedule of timed manoeuvres, used to script the demos."""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Phase:
    """One manoeuvre: what to command, what to label it, and for how long."""

    action: Callable[[], None]
    label: str
    duration_s: float


class PhaseCycle:
    """Loops through phases forever, in wall-clock time.

    The clock starts on the first call to `current`, so a demo can build the
    cycle up front and only begin timing once the drone is actually airborne.
    """

    def __init__(self, phases: Sequence[Phase]):
        if not phases:
            raise ValueError("PhaseCycle needs at least one phase")
        self._phases = list(phases)
        self._duration_s = sum(phase.duration_s for phase in self._phases)
        self._started_at: float | None = None

        self._offsets: list[float] = []
        offset = 0.0
        for phase in self._phases:
            self._offsets.append(offset)
            offset += phase.duration_s

    @property
    def duration_s(self) -> float:
        return self._duration_s

    def current(self) -> Phase:
        """The phase that is active now, restarting the cycle when it runs out."""
        if self._started_at is None:
            self._started_at = time.time()
        return self.phase_at((time.time() - self._started_at) % self._duration_s)

    def phase_at(self, elapsed_s: float) -> Phase:
        index = 0
        for candidate, offset in enumerate(self._offsets):
            if elapsed_s >= offset:
                index = candidate
        return self._phases[index]

    def run_current(self) -> str:
        """Command the active phase and return its label."""
        phase = self.current()
        phase.action()
        return phase.label

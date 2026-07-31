import pytest

from drone_landing.simulation import Phase, PhaseCycle


def noop() -> None:
    pass


def cycle_of(*durations: float) -> PhaseCycle:
    return PhaseCycle([
        Phase(action=noop, label=f"phase{index}", duration_s=duration)
        for index, duration in enumerate(durations)
    ])


def test_rejects_an_empty_schedule():
    with pytest.raises(ValueError, match="at least one phase"):
        PhaseCycle([])


def test_duration_is_the_sum_of_the_phases():
    assert cycle_of(1.0, 2.0, 3.0).duration_s == 6.0


@pytest.mark.parametrize(
    ("elapsed", "expected"),
    [
        (0.0, "phase0"),
        (0.9, "phase0"),
        (1.0, "phase1"),
        (2.9, "phase1"),
        (3.0, "phase2"),
        (5.9, "phase2"),
    ],
)
def test_phase_at_selects_the_active_phase(elapsed, expected):
    assert cycle_of(1.0, 2.0, 3.0).phase_at(elapsed).label == expected


def test_run_current_invokes_the_action_and_returns_its_label():
    calls = []
    cycle = PhaseCycle([
        Phase(action=lambda: calls.append("ran"), label="only", duration_s=1.0),
    ])

    label = cycle.run_current()

    assert label == "only"
    assert calls == ["ran"]

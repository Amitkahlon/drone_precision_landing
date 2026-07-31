"""One JSON record per run, appended to the results file as the batch proceeds."""

import statistics
from datetime import datetime, timezone

from ..scenarios import Scenario, ScenarioConfig
from ..simulation import RunResult


def build_record(
        batch_id: str,
        index: int,
        scenario: Scenario,
        result: RunResult,
        config: ScenarioConfig,
) -> dict:
    """Flatten a run into a self-contained record.

    The generating config and seed are stored alongside the outcome so any run in
    the dataset can be reproduced without reference to the batch it came from.
    """
    speeds = [waypoint.speed for waypoint in scenario.waypoints]
    return {
        "run_id": f"{batch_id}-{index:03d}",
        "batch_id": batch_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **scenario.to_dict(),
        "leg_speed_min": round(min(speeds), 4),
        "leg_speed_mean": round(statistics.fmean(speeds), 4),
        "leg_speed_max": round(max(speeds), 4),
        **result.to_dict(),
        "config": config.to_dict(),
    }


def format_record(record: dict) -> str:
    """One-line progress summary of a finished run."""
    outcome = "OK  " if record["success"] else "FAIL"
    reason = record["failure_reason"] or "landed on platform"
    error = f"{record['xy_error_m']:.3f}m" if record["xy_error_m"] is not None else "n/a"
    return (
        f"  {record['run_id']} seed={record['seed']} "
        f"wp={record['num_waypoints']} speed={record['leg_speed_mean']:.2f} m/s "
        f"t={record['duration_s']:.1f}s err={error} "
        f"{outcome} ({reason})"
    )

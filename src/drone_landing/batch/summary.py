"""Aggregate statistics over a batch of run records."""

import statistics
from collections.abc import Callable

from ..scenarios import ScenarioConfig
from ..settings import SPEED_BUCKET_COUNT


def summarize(
        batch_id: str,
        base_seed: int,
        records: list[dict],
        config: ScenarioConfig,
) -> dict:
    total = len(records)
    successes = sum(record["success"] for record in records)
    return {
        "batch_id": batch_id,
        "base_seed": base_seed,
        "total_runs": total,
        "successes": successes,
        "success_rate": round(successes / total, 4) if total else 0.0,
        "landed_but_off_platform": _count_reason(records, "off_platform"),
        "timeouts": _count_reason(records, "timeout"),
        "realign_exhausted": _count_reason(records, "realign_exhausted"),
        "diverged": _count_reason(records, "diverged"),
        "aborted": _count_reason(records, "aborted"),
        "mean_duration_s": (
            round(statistics.fmean([r["duration_s"] for r in records]), 3) if total else 0.0
        ),
        "by_waypoint_count": breakdown(records, lambda r: r["num_waypoints"]),
        "by_speed_bucket": breakdown(records, lambda r: speed_bucket(r, config)),
        "failed_seeds": [r["seed"] for r in records if not r["success"]],
    }


def breakdown(records: list[dict], key: Callable[[dict], object]) -> dict:
    """Success rate per distinct value of `key`, ordered by that value's text form."""
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(str(key(record)), []).append(record)
    return {
        name: {
            "runs": len(group),
            "successes": sum(r["success"] for r in group),
            "success_rate": round(sum(r["success"] for r in group) / len(group), 4),
        }
        for name, group in sorted(groups.items())
    }


def speed_bucket(record: dict, config: ScenarioConfig) -> str:
    """Label the fixed-width speed band a run's mean leg speed falls into."""
    span = config.max_speed - config.min_speed
    width = span / SPEED_BUCKET_COUNT if span > 0 else 1.0
    index = min(
        int((record["leg_speed_mean"] - config.min_speed) / width),
        SPEED_BUCKET_COUNT - 1,
    )
    low = config.min_speed + index * width
    return f"{low:.2f}-{low + width:.2f}"


def _count_reason(records: list[dict], reason: str) -> int:
    return sum(record["failure_reason"] == reason for record in records)

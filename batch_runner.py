"""Stress-test the landing controller across randomized scenarios.

    python batch_runner.py --runs 50                 # headless batch
    python batch_runner.py --seed 12345 --runs 1     # replay one scenario
    python batch_runner.py --seed 12345 --runs 1 --viewer
"""

import argparse
import json
import os
import random
import statistics
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from controls import ScenarioConfig, generate_scenario, run_mission

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DEFAULT_OUTPUT = os.path.join(RESULTS_DIR, "runs.jsonl")

_SPEED_BUCKETS = 4


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = _config_from_args(args)

    # Drawn from system entropy when unset, then reported so any batch stays replayable.
    base_seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2 ** 32)
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    print(f"batch {batch_id}: {args.runs} run(s), base seed {base_seed}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    records = []
    with open(args.output, "a") as f:
        for index in range(args.runs):
            seed = base_seed + index
            scenario = generate_scenario(config, seed)
            result = run_mission(
                scenario.build_mission(),
                hover_altitude=args.hover_altitude,
                timeout=args.timeout,
                viewer=args.viewer,
            )

            record = _record(batch_id, index, scenario, result, config)
            records.append(record)
            f.write(json.dumps(record) + "\n")
            f.flush()
            print(_format_run(record))

    summary = _summarize(batch_id, base_seed, records, config)
    _print_summary(summary)

    summary_path = os.path.join(RESULTS_DIR, f"summary_{batch_id}.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nresults appended to {args.output}\nsummary written to {summary_path}")
    return 0


# --- config ---

def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", "--runs", type=int, default=20, help="scenarios to run in this batch")
    parser.add_argument("--seed", type=int, default=None,
                        help="base RNG seed; run i uses seed+i, so --seed S --runs 1 replays run S")
    parser.add_argument("--bounds", type=float, nargs=2, metavar=("X", "Y"), default=(8.0, 8.0),
                        help="half-extents of the waypoint box around the launch point")
    parser.add_argument("--min-waypoints", type=int, default=3)
    parser.add_argument("--max-waypoints", type=int, default=8)
    parser.add_argument("--min-speed", type=float, default=0.5)
    parser.add_argument("--max-speed", type=float, default=2.0)
    parser.add_argument("--min-altitude", type=float, default=0.5)
    parser.add_argument("--max-altitude", type=float, default=0.5)
    parser.add_argument("--min-spacing", type=float, default=2.0,
                        help="minimum XY distance between waypoints")
    parser.add_argument("--timeout", type=float, default=120.0, help="per-run simulated seconds")
    parser.add_argument("--hover-altitude", type=float, default=3.0)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--viewer", action="store_true", help="watch each run in the passive viewer")
    return parser.parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> ScenarioConfig:
    return ScenarioConfig(
        bounds_x=args.bounds[0],
        bounds_y=args.bounds[1],
        min_waypoints=args.min_waypoints,
        max_waypoints=args.max_waypoints,
        min_speed=args.min_speed,
        max_speed=args.max_speed,
        min_altitude=args.min_altitude,
        max_altitude=args.max_altitude,
        min_spacing=args.min_spacing,
    )


# --- logging ---

def _record(batch_id: str, index: int, scenario, result, config: ScenarioConfig) -> dict:
    speeds = [wp.speed for wp in scenario.waypoints]
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


def _format_run(record: dict) -> str:
    outcome = "OK  " if record["success"] else "FAIL"
    reason = record["failure_reason"] or "landed on platform"
    error = f"{record['xy_error_m']:.3f}m" if record["xy_error_m"] is not None else "n/a"
    return (
        f"  {record['run_id']} seed={record['seed']} "
        f"wp={record['num_waypoints']} speed={record['leg_speed_mean']:.2f} m/s "
        f"t={record['duration_s']:.1f}s err={error} "
        f"{outcome} ({reason})"
    )


# --- summary ---

def _summarize(batch_id: str, base_seed: int, records: list[dict], config: ScenarioConfig) -> dict:
    total = len(records)
    successes = sum(r["success"] for r in records)
    return {
        "batch_id": batch_id,
        "base_seed": base_seed,
        "total_runs": total,
        "successes": successes,
        "success_rate": round(successes / total, 4) if total else 0.0,
        "landed_but_off_platform": sum(r["failure_reason"] == "off_platform" for r in records),
        "timeouts": sum(r["failure_reason"] == "timeout" for r in records),
        "diverged": sum(r["failure_reason"] == "diverged" for r in records),
        "aborted": sum(r["failure_reason"] == "aborted" for r in records),
        "mean_duration_s": round(statistics.fmean([r["duration_s"] for r in records]), 3) if total else 0.0,
        "by_waypoint_count": _breakdown(records, lambda r: r["num_waypoints"]),
        "by_speed_bucket": _breakdown(records, lambda r: _speed_bucket(r, config)),
        "failed_seeds": [r["seed"] for r in records if not r["success"]],
    }


def _breakdown(records: list[dict], key) -> dict:
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


def _speed_bucket(record: dict, config: ScenarioConfig) -> str:
    span = config.max_speed - config.min_speed
    width = span / _SPEED_BUCKETS if span > 0 else 1.0
    index = min(int((record["leg_speed_mean"] - config.min_speed) / width), _SPEED_BUCKETS - 1)
    low = config.min_speed + index * width
    return f"{low:.2f}-{low + width:.2f}"


def _print_summary(summary: dict) -> None:
    print(
        f"\n{summary['total_runs']} runs, {summary['successes']} succeeded "
        f"({summary['success_rate'] * 100:.1f}%), "
        f"{summary['landed_but_off_platform']} off-platform, "
        f"{summary['timeouts']} timed out, {summary['diverged']} diverged"
    )
    for title, key in (("by waypoint count", "by_waypoint_count"), ("by mean leg speed", "by_speed_bucket")):
        print(f"\n  {title}:")
        for name, stats in summary[key].items():
            print(f"    {name:>10}  {stats['successes']:>3}/{stats['runs']:<3} "
                  f"{stats['success_rate'] * 100:5.1f}%")
    if summary["failed_seeds"]:
        print(f"\n  replay a failure: python batch_runner.py --seed {summary['failed_seeds'][0]} --runs 1 --viewer")


if __name__ == "__main__":
    raise SystemExit(main())

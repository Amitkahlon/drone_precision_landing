"""Human-readable rendering of a batch summary.

These go to stdout rather than through logging: they are the command's output,
not diagnostics about it.
"""

from pathlib import Path

_BREAKDOWNS = (
    ("by waypoint count", "by_waypoint_count"),
    ("by mean leg speed", "by_speed_bucket"),
)


def print_batch_start(batch_id: str, runs: int, base_seed: int) -> None:
    print(f"batch {batch_id}: {runs} run(s), base seed {base_seed}")


def print_summary(summary: dict) -> None:
    print(
        f"\n{summary['total_runs']} runs, {summary['successes']} succeeded "
        f"({summary['success_rate'] * 100:.1f}%), "
        f"{summary['landed_but_off_platform']} off-platform, "
        f"{summary['timeouts']} timed out, {summary['diverged']} diverged"
    )
    for title, key in _BREAKDOWNS:
        print(f"\n  {title}:")
        for name, stats in summary[key].items():
            print(f"    {name:>10}  {stats['successes']:>3}/{stats['runs']:<3} "
                  f"{stats['success_rate'] * 100:5.1f}%")
    if summary["failed_seeds"]:
        print(
            f"\n  replay a failure: drone-batch "
            f"--seed {summary['failed_seeds'][0]} --runs 1 --viewer"
        )


def print_output_paths(runs_path: Path, summary_path: Path) -> None:
    print(f"\nresults appended to {runs_path}\nsummary written to {summary_path}")

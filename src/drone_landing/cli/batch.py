"""Stress-test the landing controller across randomized scenarios.

    drone-batch --runs 50                 # headless batch
    drone-batch --seed 12345 --runs 1     # replay one scenario
    drone-batch --seed 12345 --runs 1 --viewer
"""

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from ..batch import build_record, format_record, summarize
from ..batch import report
from ..scenarios import ScenarioConfig, generate_scenario
from ..settings import (
    DEFAULT_HOVER_ALTITUDE_M,
    DEFAULT_RUNS_FILE,
    DEFAULT_TIMEOUT_S,
)
from ..simulation import run_mission


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = _config_from_args(args)

    # Drawn from system entropy when unset, then reported so any batch stays replayable.
    base_seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2 ** 32)
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    report.print_batch_start(batch_id, args.runs, base_seed)

    runs_path = Path(args.output).resolve()
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    records = _run_batch(args, config, batch_id, base_seed, runs_path)

    summary = summarize(batch_id, base_seed, records, config)
    report.print_summary(summary)

    summary_path = runs_path.parent / f"summary_{batch_id}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    report.print_output_paths(runs_path, summary_path)
    return 0


def _run_batch(
        args: argparse.Namespace,
        config: ScenarioConfig,
        batch_id: str,
        base_seed: int,
        runs_path: Path,
) -> list[dict]:
    """Fly every scenario, appending each record as soon as it finishes.

    Flushed per run so a long batch that is interrupted still leaves usable data.
    """
    records = []
    with open(runs_path, "a") as results_file:
        for index in range(args.runs):
            scenario = generate_scenario(config, base_seed + index)
            result = run_mission(
                scenario.build_mission(),
                hover_altitude=args.hover_altitude,
                timeout=args.timeout,
                viewer=args.viewer,
            )

            record = build_record(batch_id, index, scenario, result, config)
            records.append(record)
            results_file.write(json.dumps(record) + "\n")
            results_file.flush()
            print(format_record(record))
    return records


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-n", "--runs", type=int, default=20,
                        help="scenarios to run in this batch")
    parser.add_argument("--seed", type=int, default=None,
                        help="base RNG seed; run i uses seed+i, so --seed S --runs 1 replays run S")
    parser.add_argument("--bounds", type=float, nargs=2, metavar=("X", "Y"),
                        default=(ScenarioConfig.default("bounds_x"),
                                 ScenarioConfig.default("bounds_y")),
                        help="half-extents of the waypoint box around the launch point")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S,
                        help="per-run simulated seconds")
    parser.add_argument("--hover-altitude", type=float, default=DEFAULT_HOVER_ALTITUDE_M)
    parser.add_argument("--output", default=str(DEFAULT_RUNS_FILE))
    parser.add_argument("--viewer", action="store_true",
                        help="watch each run in the passive viewer")

    _add_config_args(parser)
    return parser.parse_args(argv)


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    """Add the flags that map one-to-one onto ScenarioConfig fields.

    Defaults are read off the dataclass so the CLI and the programmatic API
    cannot drift apart.
    """
    fields = (
        ("min-waypoints", int, None),
        ("max-waypoints", int, None),
        ("min-speed", float, None),
        ("max-speed", float, None),
        ("min-altitude", float, None),
        ("max-altitude", float, None),
        ("min-spacing", float, "minimum XY distance between waypoints"),
    )
    for flag, value_type, help_text in fields:
        parser.add_argument(
            f"--{flag}",
            type=value_type,
            default=ScenarioConfig.default(flag.replace("-", "_")),
            help=help_text,
        )


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


if __name__ == "__main__":
    raise SystemExit(main())

import json

from drone_landing import ScenarioConfig, generate_scenario
from drone_landing.batch import build_record, format_record, speed_bucket, summarize
from drone_landing.cli import batch as batch_cli
from drone_landing.simulation import RunResult


def make_result(success: bool, failure_reason: str | None, duration_s: float = 10.0) -> RunResult:
    return RunResult(
        landed=success or failure_reason == "off_platform",
        on_platform=success,
        success=success,
        failure_reason=failure_reason,
        final_state="GROUNDED" if success else "LANDING",
        duration_s=duration_s,
        steps=int(duration_s * 500),
        xy_error_m=0.05 if success else 2.0,
        drone_position=[0.0, 0.0, 0.6],
        platform_position=[0.0, 0.0, 0.5],
    )


def make_record(seed: int, success: bool, failure_reason: str | None = None, **overrides) -> dict:
    config = ScenarioConfig()
    scenario = generate_scenario(config, seed)
    record = build_record("BATCH", 0, scenario, make_result(success, failure_reason), config)
    record.update(overrides)
    return record


def test_record_carries_seed_scenario_result_and_config():
    record = make_record(42, success=True)

    assert record["run_id"] == "BATCH-000"
    assert record["seed"] == 42
    assert record["success"] is True
    assert record["config"] == ScenarioConfig().to_dict()
    assert record["leg_speed_min"] <= record["leg_speed_mean"] <= record["leg_speed_max"]


def test_record_serialises_to_json():
    record = make_record(7, success=False, failure_reason="off_platform")

    assert json.loads(json.dumps(record))["failure_reason"] == "off_platform"


def test_format_record_reports_success_and_failure_differently():
    success_line = format_record(make_record(42, success=True))
    failure_line = format_record(make_record(42, success=False, failure_reason="timeout"))

    assert "OK" in success_line and "landed on platform" in success_line
    assert "FAIL" in failure_line and "timeout" in failure_line


def test_format_record_handles_a_missing_error():
    line = format_record(make_record(42, success=False, failure_reason="diverged", xy_error_m=None))

    assert "err=n/a" in line


def test_summary_counts_each_failure_mode():
    records = [
        make_record(1, success=True),
        make_record(2, success=False, failure_reason="off_platform"),
        make_record(3, success=False, failure_reason="timeout"),
        make_record(4, success=False, failure_reason="diverged"),
        make_record(5, success=False, failure_reason="aborted"),
        make_record(6, success=False, failure_reason="realign_exhausted"),
    ]

    summary = summarize("BATCH", 1, records, ScenarioConfig())

    assert summary["total_runs"] == 6
    assert summary["successes"] == 1
    assert summary["success_rate"] == 0.1667
    assert summary["landed_but_off_platform"] == 1
    assert summary["timeouts"] == 1
    assert summary["diverged"] == 1
    assert summary["aborted"] == 1
    assert summary["realign_exhausted"] == 1
    assert summary["failed_seeds"] == [r["seed"] for r in records[1:]]


def test_summary_of_an_empty_batch_does_not_divide_by_zero():
    summary = summarize("BATCH", 1, [], ScenarioConfig())

    assert summary["total_runs"] == 0
    assert summary["success_rate"] == 0.0
    assert summary["mean_duration_s"] == 0.0


def test_speed_buckets_span_the_configured_range():
    config = ScenarioConfig(min_speed=0.5, max_speed=2.5)

    assert speed_bucket({"leg_speed_mean": 0.5}, config) == "0.50-1.00"
    assert speed_bucket({"leg_speed_mean": 1.2}, config) == "1.00-1.50"
    assert speed_bucket({"leg_speed_mean": 2.5}, config) == "2.00-2.50"


def test_speed_bucket_survives_a_zero_width_range():
    config = ScenarioConfig(min_speed=1.0, max_speed=1.0)

    assert speed_bucket({"leg_speed_mean": 1.0}, config) == "1.00-2.00"


def test_cli_defaults_come_from_the_scenario_dataclass():
    args = batch_cli._parse_args([])
    defaults = ScenarioConfig()

    assert batch_cli._config_from_args(args) == defaults


def test_cli_flags_override_the_dataclass_defaults():
    args = batch_cli._parse_args(["--min-speed", "1.25", "--bounds", "3", "4"])

    config = batch_cli._config_from_args(args)

    assert config.min_speed == 1.25
    assert (config.bounds_x, config.bounds_y) == (3.0, 4.0)


def test_batch_writes_records_and_summary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "out" / "runs.jsonl"

    exit_code = batch_cli.main(["--seed", "42", "--runs", "2", "--output", str(output)])

    assert exit_code == 0
    lines = output.read_text().strip().splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["seed"] for line in lines] == [42, 43]

    summaries = list(output.parent.glob("summary_*.json"))
    assert len(summaries) == 1
    assert json.loads(summaries[0].read_text())["total_runs"] == 2


def test_batch_appends_to_an_existing_results_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "runs.jsonl"

    batch_cli.main(["--seed", "42", "--runs", "1", "--output", str(output)])
    batch_cli.main(["--seed", "99", "--runs", "1", "--output", str(output)])

    assert len(output.read_text().strip().splitlines()) == 2

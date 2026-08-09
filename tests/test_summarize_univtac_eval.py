from pathlib import Path

from summarize_univtac_eval import parse_log, summarize


def test_parse_and_merge_sharded_univtac_logs(tmp_path: Path):
    first = tmp_path / "2026-01-01_00:00:00"
    second = tmp_path / "2026-01-01_00:01:00"
    first.mkdir()
    second.mkdir()
    (first / "log.log").write_text(
        "[2026-01-01 00:00:01] [1  ] Seed 10 success after 1.25 s.\n"
        "steps: 20   , actions: 3    .\n"
        "[2026-01-01 00:00:02] [2  ] Seed 11 failed after 2.50 s.\n"
        "steps: 300  , actions: 300  .\n",
        encoding="utf-8",
    )
    (second / "log.log").write_text(
        "[2026-01-01 00:01:01] [1  ] Seed 12 occurred exception: example\n"
        "[2026-01-01 00:01:02] [1  ] Seed 12 success after 1.00 s.\n"
        "steps: 10   , actions: 2    .\n",
        encoding="utf-8",
    )

    records, errors = parse_log(second / "log.log")
    assert records[0]["seed"] == 12
    assert errors[0]["error"] == "example"

    result = summarize(tmp_path, 10, 13)
    assert result["completed_episodes"] == 3
    assert result["successes"] == 2
    assert result["failures"] == 1
    assert result["success_rate_percent"] == 200 / 3
    assert result["missing_seeds"] == [13]
    assert not result["complete"]

import json
from datetime import date

from kalshi_quant.monitoring.reports import build_period_report, format_period_report, write_weekly_report


def write_record(log_dir, day: date, record: dict):
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{day.isoformat()}.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def test_build_period_report_aggregates_settlements(tmp_path):
    day = date(2024, 1, 1)
    write_record(tmp_path, day, {"event_type": "settlement", "market_ticker": "A", "won": True, "pnl": 5.0})
    write_record(tmp_path, day, {"event_type": "settlement", "market_ticker": "B", "won": False, "pnl": -3.0})

    report = build_period_report(tmp_path, day, day)
    assert report.n_settlements == 2
    assert report.total_pnl_dollars == 2.0
    assert report.win_rate == 0.5


def test_build_period_report_empty_window(tmp_path):
    report = build_period_report(tmp_path, date(2024, 1, 1), date(2024, 1, 1))
    assert report.n_settlements == 0
    assert report.total_pnl_dollars == 0.0
    assert report.calibration is None


def test_build_period_report_matches_decisions_to_settlements_for_calibration(tmp_path):
    day = date(2024, 1, 1)
    for i in range(10):
        won = i % 2 == 0
        prob = 0.9 if won else 0.1
        write_record(tmp_path, day, {
            "event_type": "decision", "market_ticker": f"M{i}", "estimate": {"probability": prob},
        })
        write_record(tmp_path, day, {
            "event_type": "settlement", "market_ticker": f"M{i}", "won": won, "pnl": 1.0 if won else -1.0,
        })
    report = build_period_report(tmp_path, day, day)
    assert report.calibration is not None
    assert report.calibration.brier_score < 0.1  # predictions matched outcomes well


def test_format_period_report_contains_key_fields(tmp_path):
    day = date(2024, 1, 1)
    write_record(tmp_path, day, {"event_type": "settlement", "market_ticker": "A", "won": True, "pnl": 5.0})
    report = build_period_report(tmp_path, day, day)
    text = format_period_report(report)
    assert "Total P&L" in text
    assert "Win rate" in text


def test_write_weekly_report_creates_file(tmp_path):
    log_dir = tmp_path / "logs"
    out_dir = tmp_path / "reports"
    day = date(2024, 1, 5)
    write_record(log_dir, day, {"event_type": "settlement", "market_ticker": "A", "won": True, "pnl": 1.0})
    path = write_weekly_report(log_dir, out_dir, as_of=day)
    assert path.exists()
    assert "Total P&L" in path.read_text()

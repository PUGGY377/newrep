import json
from datetime import datetime

from kalshi_quant.execution.audit import AuditLogger


def test_log_creates_daily_file(tmp_path):
    logger = AuditLogger(tmp_path)
    logger.log("test_event", datetime(2024, 3, 15, 10, 0), foo="bar")
    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    assert files[0].name == "2024-03-15.jsonl"


def test_log_entries_are_valid_json_lines(tmp_path):
    logger = AuditLogger(tmp_path)
    logger.log("a", datetime(2024, 1, 1), x=1)
    logger.log("b", datetime(2024, 1, 1), y=2)
    lines = (tmp_path / "2024-01-01.jsonl").read_text().strip().split("\n")
    assert len(lines) == 2
    parsed = [json.loads(l) for l in lines]
    assert parsed[0]["event_type"] == "a"
    assert parsed[1]["event_type"] == "b"


def test_log_serializes_datetimes_and_enums(tmp_path):
    from kalshi_quant.data.models import Side
    logger = AuditLogger(tmp_path)
    logger.log("with_types", datetime(2024, 1, 1), side=Side.YES, nested={"when": datetime(2024, 1, 2)})
    line = (tmp_path / "2024-01-01.jsonl").read_text().strip()
    parsed = json.loads(line)
    assert parsed["side"] == "yes"
    assert parsed["nested"]["when"] == "2024-01-02T00:00:00"

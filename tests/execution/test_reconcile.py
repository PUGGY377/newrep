from kalshi_quant.execution.audit import AuditLogger
from kalshi_quant.execution.reconcile import reconcile


class FakeKalshiClient:
    def __init__(self, positions):
        self._positions = positions

    def get_positions(self):
        return self._positions


def test_reconcile_finds_no_discrepancy_when_matching(tmp_path):
    client = FakeKalshiClient([{"ticker": "A", "position": 10}])
    audit = AuditLogger(tmp_path)
    discrepancies = reconcile({"A": 10}, client, audit)
    assert discrepancies == []


def test_reconcile_flags_mismatch(tmp_path):
    client = FakeKalshiClient([{"ticker": "A", "position": 5}])
    audit = AuditLogger(tmp_path)
    discrepancies = reconcile({"A": 10}, client, audit)
    assert len(discrepancies) == 1
    assert discrepancies[0].local_position == 10
    assert discrepancies[0].remote_position == 5
    logged = list(tmp_path.glob("*.jsonl"))
    assert len(logged) == 1


def test_reconcile_flags_position_missing_locally():
    client = FakeKalshiClient([{"ticker": "B", "position": 3}])
    from kalshi_quant.execution.audit import AuditLogger as AL
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        discrepancies = reconcile({}, client, AL(d))
        assert len(discrepancies) == 1
        assert discrepancies[0].market_ticker == "B"

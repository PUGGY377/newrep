from datetime import datetime

from kalshi_quant.monitoring.alerts import Alert, AlertManager, AlertSeverity


def test_model_drift_detected_on_large_shift():
    captured = []
    manager = AlertManager(notify=captured.append)
    historical = [0.5] * 50
    latest = [0.9] * 5
    alert = manager.check_model_output_drift("m", latest, historical, z_score_threshold=3.0)
    assert alert is not None
    assert alert.severity == AlertSeverity.WARNING
    assert captured == [alert]


def test_model_drift_not_triggered_on_small_shift():
    manager = AlertManager(notify=lambda a: None)
    historical = [0.5 + 0.01 * (i % 3) for i in range(50)]
    latest = [0.51, 0.52]
    alert = manager.check_model_output_drift("m", latest, historical)
    assert alert is None


def test_model_drift_skipped_with_insufficient_history():
    manager = AlertManager(notify=lambda a: None)
    alert = manager.check_model_output_drift("m", [0.9], [0.5, 0.5])
    assert alert is None


def test_api_failure_triggers_after_threshold():
    captured = []
    manager = AlertManager(notify=captured.append)
    for _ in range(2):
        result = manager.record_api_failure("endpoint", "timeout")
        assert result is None
    alert = manager.record_api_failure("endpoint", "timeout")
    assert alert is not None
    assert alert.severity == AlertSeverity.CRITICAL
    assert len(captured) == 1


def test_api_success_resets_failure_count():
    manager = AlertManager(notify=lambda a: None)
    manager.record_api_failure("e", "err")
    manager.record_api_failure("e", "err")
    manager.record_api_success()
    result = manager.record_api_failure("e", "err")
    assert result is None  # counter was reset, so only 1 failure so far


def test_drawdown_breach_alert():
    captured = []
    manager = AlertManager(notify=captured.append)
    alert = manager.check_drawdown_breach(0.45, 0.40)
    assert alert is not None
    assert alert.category == "drawdown_breach"
    assert captured == [alert]


def test_drawdown_below_threshold_no_alert():
    manager = AlertManager(notify=lambda a: None)
    alert = manager.check_drawdown_breach(0.10, 0.40)
    assert alert is None

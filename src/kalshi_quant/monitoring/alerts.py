"""Anomaly alerting.

Three alert classes, matching the spec: model output drifting from its
own recent behavior, API failures, and drawdown/circuit-breaker
breaches. `AlertManager` is decoupled from how alerts are delivered --
`notify` defaults to structured logging, and can be swapped for a real
notification channel (email, Slack webhook, SMS) without touching the
detection logic.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable

import structlog

logger = structlog.get_logger(__name__)


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class Alert:
    category: str
    severity: AlertSeverity
    message: str
    when: datetime
    context: dict = field(default_factory=dict)


def _default_notify(alert: Alert) -> None:
    log_fn = {
        AlertSeverity.INFO: logger.info, AlertSeverity.WARNING: logger.warning,
        AlertSeverity.CRITICAL: logger.error,
    }[alert.severity]
    log_fn("alert", category=alert.category, message=alert.message, **alert.context)


class AlertManager:
    def __init__(self, notify: Callable[[Alert], None] = _default_notify):
        self._notify = notify
        self._api_failure_count = 0

    def check_model_output_drift(
        self, model_name: str, latest_probabilities: list[float], historical_probabilities: list[float],
        z_score_threshold: float = 3.0, when: datetime | None = None,
    ) -> Alert | None:
        """Flags when the mean of `latest_probabilities` (e.g. this week's
        predictions) is an outlier relative to the historical distribution
        of predictions -- a proxy for "the model started behaving very
        differently," which is worth a human look regardless of cause
        (data pipeline break, regime change, a bug in a recent deploy)."""
        if len(historical_probabilities) < 10 or not latest_probabilities:
            return None
        hist_mean = statistics.mean(historical_probabilities)
        hist_stdev = statistics.pstdev(historical_probabilities) or 1e-6
        latest_mean = statistics.mean(latest_probabilities)
        z = abs(latest_mean - hist_mean) / hist_stdev
        if z >= z_score_threshold:
            alert = Alert(
                category="model_drift", severity=AlertSeverity.WARNING,
                message=(
                    f"{model_name}: recent mean prediction {latest_mean:.3f} is {z:.1f} std devs "
                    f"from historical mean {hist_mean:.3f}"
                ),
                when=when or datetime.utcnow(),
                context={"model_name": model_name, "z_score": z, "latest_mean": latest_mean,
                         "historical_mean": hist_mean},
            )
            self._notify(alert)
            return alert
        return None

    def record_api_failure(self, endpoint: str, error: str, when: datetime | None = None,
                            consecutive_failure_threshold: int = 3) -> Alert | None:
        self._api_failure_count += 1
        if self._api_failure_count >= consecutive_failure_threshold:
            alert = Alert(
                category="api_failure", severity=AlertSeverity.CRITICAL,
                message=f"{self._api_failure_count} consecutive API failures (latest: {endpoint}: {error})",
                when=when or datetime.utcnow(), context={"endpoint": endpoint, "error": error},
            )
            self._notify(alert)
            return alert
        return None

    def record_api_success(self) -> None:
        self._api_failure_count = 0

    def check_drawdown_breach(
        self, current_drawdown_fraction: float, hard_stop_fraction: float, when: datetime | None = None,
    ) -> Alert | None:
        if current_drawdown_fraction >= hard_stop_fraction:
            alert = Alert(
                category="drawdown_breach", severity=AlertSeverity.CRITICAL,
                message=(
                    f"Drawdown {current_drawdown_fraction:.1%} has reached the hard-stop threshold "
                    f"{hard_stop_fraction:.1%}; circuit breaker should be halting new positions."
                ),
                when=when or datetime.utcnow(),
                context={"drawdown": current_drawdown_fraction, "threshold": hard_stop_fraction},
            )
            self._notify(alert)
            return alert
        return None

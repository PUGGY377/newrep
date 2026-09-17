"""Reconciliation: compare local position state against Kalshi's actual
account state (live trading) and flag discrepancies rather than silently
trusting local bookkeeping. In paper mode there is no external account to
reconcile against -- this module is exercised once live trading exists,
but the interface is defined now so live.py has somewhere to plug into.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from kalshi_quant.data.kalshi_client import KalshiClient, utc_now
from kalshi_quant.execution.audit import AuditLogger


@dataclass(frozen=True)
class Discrepancy:
    market_ticker: str
    local_position: int
    remote_position: int

    def as_dict(self) -> dict:
        return {
            "market_ticker": self.market_ticker, "local_position": self.local_position,
            "remote_position": self.remote_position,
        }


def reconcile(
    local_positions: dict[str, int], kalshi_client: KalshiClient, audit_logger: AuditLogger,
) -> list[Discrepancy]:
    remote_positions = {
        raw.get("ticker"): int(raw.get("position", 0)) for raw in kalshi_client.get_positions()
    }
    tickers = set(local_positions) | set(remote_positions)
    discrepancies = []
    for ticker in tickers:
        local = local_positions.get(ticker, 0)
        remote = remote_positions.get(ticker, 0)
        if local != remote:
            discrepancies.append(Discrepancy(market_ticker=ticker, local_position=local, remote_position=remote))

    if discrepancies:
        audit_logger.log_reconciliation_discrepancy(utc_now(), [d.as_dict() for d in discrepancies])

    return discrepancies

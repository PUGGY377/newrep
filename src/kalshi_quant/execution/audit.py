"""Full decision audit log.

Every trade decision -- forecast inputs, computed edge, chosen position
size, order sent, and fill received -- is written as one JSON line per
event to a daily append-only file. The goal stated in the spec is literal:
being able to reconstruct exactly why any trade happened, after the fact,
without relying on memory of what the code did at the time.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime | date):
        return obj.isoformat()
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if hasattr(obj, "value"):  # Enum
        return obj.value
    return str(obj)


class AuditLogger:
    def __init__(self, log_dir: str | Path = "audit_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, when: datetime) -> Path:
        return self.log_dir / f"{when.date().isoformat()}.jsonl"

    def log(self, event_type: str, when: datetime, **fields: Any) -> None:
        record = {"event_type": event_type, "timestamp": when.isoformat(), **fields}
        line = json.dumps(record, default=_json_default)
        with open(self._path_for(when), "a") as f:
            f.write(line + "\n")

    def log_decision(
        self, when: datetime, market_ticker: str, estimate: Any, edge: Any, position_size: Any,
    ) -> None:
        self.log(
            "decision", when, market_ticker=market_ticker, estimate=estimate, edge=edge,
            position_size=position_size,
        )

    def log_order_submitted(self, when: datetime, order: Any) -> None:
        self.log("order_submitted", when, order=order)

    def log_fill(self, when: datetime, fill: Any) -> None:
        self.log("fill", when, fill=fill)

    def log_safety_rail_block(self, when: datetime, reason: str, context: dict[str, Any]) -> None:
        self.log("safety_rail_block", when, reason=reason, context=context)

    def log_reconciliation_discrepancy(self, when: datetime, discrepancies: list[dict[str, Any]]) -> None:
        self.log("reconciliation_discrepancy", when, discrepancies=discrepancies)

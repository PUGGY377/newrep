"""Periodic (default: weekly) auto-generated model/strategy performance
report, built from the execution layer's audit log -- the same JSONL
files execution/audit.py writes, read back rather than duplicated into a
separate metrics store. This keeps "what actually happened" as the single
source of truth for reporting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

from kalshi_quant.forecasting.calibration import CalibrationReport, evaluate_calibration


@dataclass(frozen=True)
class PeriodReport:
    period_start: date
    period_end: date
    n_settlements: int
    total_pnl_dollars: float
    win_rate: float
    n_decisions_logged: int
    calibration: CalibrationReport | None


def _iter_audit_records(log_dir: Path, start: date, end: date):
    d = start
    while d <= end:
        path = log_dir / f"{d.isoformat()}.jsonl"
        if path.exists():
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
        d += timedelta(days=1)


def build_period_report(log_dir: str | Path, period_start: date, period_end: date) -> PeriodReport:
    log_dir = Path(log_dir)
    settlements = []
    decisions = []
    for record in _iter_audit_records(log_dir, period_start, period_end):
        if record.get("event_type") == "settlement":
            settlements.append(record)
        elif record.get("event_type") == "decision":
            decisions.append(record)

    total_pnl = sum(r.get("pnl", 0.0) for r in settlements)
    wins = sum(1 for r in settlements if r.get("won"))
    win_rate = wins / len(settlements) if settlements else 0.0

    calibration = None
    # Calibration requires matching each decision's forecast probability to
    # its eventual settlement outcome by market_ticker -- only decisions
    # whose market actually settled within the window can be scored here.
    settlement_by_ticker = {r["market_ticker"]: r for r in settlements if "market_ticker" in r}
    preds, outcomes = [], []
    for d in decisions:
        ticker = d.get("market_ticker")
        estimate = d.get("estimate", {})
        prob = estimate.get("probability") if isinstance(estimate, dict) else None
        settlement = settlement_by_ticker.get(ticker)
        if prob is not None and settlement is not None:
            preds.append(prob)
            outcomes.append(1.0 if settlement.get("won") else 0.0)
    if len(preds) >= 5:
        calibration = evaluate_calibration(np.array(preds), np.array(outcomes))

    return PeriodReport(
        period_start=period_start, period_end=period_end, n_settlements=len(settlements),
        total_pnl_dollars=total_pnl, win_rate=win_rate, n_decisions_logged=len(decisions),
        calibration=calibration,
    )


def format_period_report(report: PeriodReport) -> str:
    lines = [
        f"=== Performance Report: {report.period_start} to {report.period_end} ===",
        f"Settlements:  {report.n_settlements}",
        f"Total P&L:    ${report.total_pnl_dollars:+.2f}",
        f"Win rate:     {report.win_rate:.1%}",
        f"Decisions logged: {report.n_decisions_logged}",
    ]
    if report.calibration:
        lines.append(f"Brier score:  {report.calibration.brier_score:.4f}")
        lines.append(f"Log loss:     {report.calibration.log_loss_value:.4f}")
    else:
        lines.append("Calibration:  insufficient matched decision/settlement data")
    return "\n".join(lines)


def write_weekly_report(log_dir: str | Path, output_dir: str | Path, as_of: date | None = None) -> Path:
    as_of = as_of or datetime.utcnow().date()
    period_start = as_of - timedelta(days=7)
    report = build_period_report(log_dir, period_start, as_of)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"weekly_report_{as_of.isoformat()}.txt"
    out_path.write_text(format_period_report(report))
    return out_path

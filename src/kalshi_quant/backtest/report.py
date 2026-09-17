"""Backtest reporting: equity curve stats, drawdown, a risk-adjusted
return metric suited to skewed binary-outcome bets, calibration, and win
rate.

Why not Sharpe: Sharpe assumes returns are roughly normal and penalizes
upside and downside volatility symmetrically. A portfolio of binary bets
has genuinely skewed, non-normal per-trade returns (a string of small
wins punctuated by occasional full losses, or vice versa) -- Sharpe on
that distribution is a well-known way to make a strategy with fat left
tail risk look better than it is. Calmar ratio (return / max drawdown)
and the empirical risk-of-ruin framing (same mu/sigma-of-log-growth model
as risk/circuit_breaker.py) are both drawdown-based, which is the
quantity that actually matters for "can this blow up the account."
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from kalshi_quant.backtest.engine import BacktestResult
from kalshi_quant.forecasting.calibration import CalibrationReport, evaluate_calibration


@dataclass(frozen=True)
class BacktestReport:
    starting_capital: float
    final_bankroll: float
    total_return_pct: float
    max_drawdown_pct: float
    calmar_ratio: float
    win_rate: float
    n_trades: int
    avg_net_edge_captured: float
    mean_log_growth_per_trade: float
    log_growth_volatility: float
    calibration: CalibrationReport | None
    halted: bool
    halt_reason: str


def max_drawdown(equity_curve: list[tuple[object, float]]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0][1]
    worst = 0.0
    for _, value in equity_curve:
        peak = max(peak, value)
        if peak > 0:
            worst = max(worst, 1 - value / peak)
    return worst


def build_report(result: BacktestResult, starting_capital: float) -> BacktestReport:
    final_bankroll = result.final_bankroll if result.equity_curve else starting_capital
    total_return_pct = (final_bankroll / starting_capital - 1.0) * 100 if starting_capital > 0 else 0.0
    mdd = max_drawdown(result.equity_curve)
    calmar = (total_return_pct / 100) / mdd if mdd > 1e-9 else float("inf") if total_return_pct > 0 else 0.0

    # Log growth is computed on consecutive EQUITY CURVE points (i.e.
    # relative to total bankroll before/after each settlement), not
    # relative to each trade's own stake -- a fractionally-sized bet that
    # loses in full should register as a small negative bankroll return,
    # not as "wealth went to zero." See risk/circuit_breaker.py for the
    # same principle applied to the risk-of-ruin halt trigger.
    log_growths = []
    for (_, before), (_, after) in zip(result.equity_curve, result.equity_curve[1:]):
        if before <= 0:
            continue
        wealth_multiple = after / before
        log_growths.append(math.log(max(wealth_multiple, 1e-6)))
    mean_log_growth = float(np.mean(log_growths)) if log_growths else 0.0
    log_growth_vol = float(np.std(log_growths, ddof=1)) if len(log_growths) > 1 else 0.0

    avg_net_edge = float(np.mean([
        t.model_probability - t.entry_price for t in result.trades
    ])) if result.trades else 0.0

    calibration = None
    if result.trades:
        preds = np.array([t.model_probability for t in result.trades])
        won = np.array([
            1.0 if (t.outcome_yes if t.side.value == "yes" else not t.outcome_yes) else 0.0
            for t in result.trades
        ])
        if len(preds) >= 5:
            calibration = evaluate_calibration(preds, won)

    return BacktestReport(
        starting_capital=starting_capital, final_bankroll=final_bankroll,
        total_return_pct=total_return_pct, max_drawdown_pct=mdd * 100, calmar_ratio=calmar,
        win_rate=result.win_rate, n_trades=len(result.trades), avg_net_edge_captured=avg_net_edge,
        mean_log_growth_per_trade=mean_log_growth, log_growth_volatility=log_growth_vol,
        calibration=calibration, halted=result.halted_at is not None, halt_reason=result.halt_reason,
    )


def format_report_text(report: BacktestReport) -> str:
    lines = [
        "=== Backtest Report ===",
        f"Starting capital:      ${report.starting_capital:,.2f}",
        f"Final bankroll:        ${report.final_bankroll:,.2f}",
        f"Total return:          {report.total_return_pct:+.2f}%",
        f"Max drawdown:          {report.max_drawdown_pct:.2f}%",
        f"Calmar ratio:          {report.calmar_ratio:.3f}",
        f"Trades:                {report.n_trades}",
        f"Win rate:              {report.win_rate:.1%}",
        f"Avg net edge captured: {report.avg_net_edge_captured:+.4f}",
        f"Mean log growth/trade: {report.mean_log_growth_per_trade:+.5f}",
        f"Log growth volatility: {report.log_growth_volatility:.5f}",
    ]
    if report.calibration:
        lines += [
            f"Brier score:           {report.calibration.brier_score:.4f}",
            f"Log loss:              {report.calibration.log_loss_value:.4f}",
        ]
    if report.halted:
        lines.append(f"CIRCUIT BREAKER TRIPPED: {report.halt_reason}")
    return "\n".join(lines)

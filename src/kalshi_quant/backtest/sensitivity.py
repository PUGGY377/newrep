"""Sensitivity analysis: how fragile is the strategy to its own assumed
parameters?

Re-runs `simulate` (cheap and pure -- see backtest/engine.py) across a
grid of edge threshold, Kelly fraction cap, and fee-multiplier scenarios
against the SAME set of precomputed TradeCandidates, so this never needs
to re-run forecasting or re-touch the data layer. A strategy whose
final bankroll swings wildly across a modest grid around the chosen
defaults is a strategy whose backtest number is not to be trusted at
face value.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from kalshi_quant.backtest.engine import SimParams, TradeCandidate, simulate
from kalshi_quant.backtest.report import build_report


@dataclass(frozen=True)
class SensitivityPoint:
    min_edge_after_fees: float
    max_kelly_fraction: float
    fee_multiplier_scale: float
    final_bankroll: float
    max_drawdown_pct: float
    total_return_pct: float
    n_trades: int


def run_sensitivity_grid(
    candidates: list[TradeCandidate],
    base_params: SimParams,
    edge_thresholds: list[float],
    kelly_fraction_caps: list[float],
    fee_multiplier_scales: list[float],
) -> list[SensitivityPoint]:
    points: list[SensitivityPoint] = []
    for edge_threshold in edge_thresholds:
        for kelly_cap in kelly_fraction_caps:
            for fee_scale in fee_multiplier_scales:
                params = replace(
                    base_params, min_edge_after_fees=edge_threshold, max_kelly_fraction=kelly_cap,
                    fee_multiplier_scale=fee_scale,
                )
                result = simulate(candidates, params)
                report = build_report(result, base_params.starting_capital)
                points.append(
                    SensitivityPoint(
                        min_edge_after_fees=edge_threshold, max_kelly_fraction=kelly_cap,
                        fee_multiplier_scale=fee_scale, final_bankroll=report.final_bankroll,
                        max_drawdown_pct=report.max_drawdown_pct, total_return_pct=report.total_return_pct,
                        n_trades=report.n_trades,
                    )
                )
    return points


def format_sensitivity_table(points: list[SensitivityPoint]) -> str:
    header = f"{'edge_thr':>9} {'kelly_cap':>10} {'fee_scale':>10} {'trades':>7} {'return%':>9} {'maxDD%':>8}"
    lines = [header]
    for p in points:
        lines.append(
            f"{p.min_edge_after_fees:>9.3f} {p.max_kelly_fraction:>10.2f} {p.fee_multiplier_scale:>10.2f} "
            f"{p.n_trades:>7d} {p.total_return_pct:>9.2f} {p.max_drawdown_pct:>8.2f}"
        )
    return "\n".join(lines)

#!/usr/bin/env python
"""Run the full backtest pipeline against a local DataStore and print a
report plus a sensitivity grid.

Usage:
    python scripts/run_backtest.py --db-path synthetic_demo.duckdb --event-prefix KXNFLGAME
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kalshi_quant.backtest.engine import SimParams, generate_candidates, simulate
from kalshi_quant.backtest.report import build_report, format_report_text
from kalshi_quant.backtest.sensitivity import format_sensitivity_table, run_sensitivity_grid
from kalshi_quant.config import load_config
from kalshi_quant.data.connectors.sports import SportsDataConnector
from kalshi_quant.data.store import DataStore
from kalshi_quant.forecasting.sports_elo import SportsEloForecaster
from kalshi_quant.risk.portfolio import PortfolioConstraints
from kalshi_quant.signal.correlation import CorrelationGraph


@click.command()
@click.option("--db-path", default="kalshi_quant.duckdb")
@click.option("--event-prefix", default="KXNFLGAME", help="event_ticker prefix to filter markets on")
@click.option("--sensitivity/--no-sensitivity", default=True)
def main(db_path: str, event_prefix: str, sensitivity: bool) -> None:
    config = load_config()
    store = DataStore(db_path)

    markets = store.get_settled_markets()
    markets = [m for m in markets if m.event_ticker.startswith(event_prefix)]
    if not markets:
        click.echo(f"No settled markets found with event_ticker prefix {event_prefix!r} in {db_path}")
        return
    click.echo(f"Loaded {len(markets)} settled markets.")

    connector = SportsDataConnector()
    forecaster = SportsEloForecaster()
    graph = CorrelationGraph()
    for m in markets:
        graph.register_event_group(m.event_ticker, [m.ticker])

    wf = config["backtest"]["walk_forward"]
    candidates = generate_candidates(
        store, forecaster, markets, connector.fetch_point_in_time, graph, category="sports",
        decision_offset_before_close=timedelta(hours=1),
        train_window_days=wf["train_window_days"], step_days=wf["step_days"],
    )
    click.echo(f"Generated {len(candidates)} trade candidates.")

    risk_cfg = config["risk"]
    params = SimParams(
        min_edge_after_fees=config["signal"]["min_edge_after_fees"],
        min_confidence=config["signal"]["min_confidence"],
        kelly_shrinkage_k=risk_cfg["kelly"]["shrinkage_k"],
        max_kelly_fraction=risk_cfg["kelly"]["max_kelly_fraction"],
        portfolio_constraints=PortfolioConstraints(
            target_portfolio_variance_frac=risk_cfg["portfolio"]["target_portfolio_variance_frac"],
            max_fraction_per_market=risk_cfg["portfolio"]["max_fraction_per_market"],
            max_fraction_per_category=risk_cfg["portfolio"]["max_fraction_per_category"],
            max_fraction_total_exposure=risk_cfg["portfolio"]["max_fraction_total_exposure"],
        ),
        max_orderbook_depth_fraction=risk_cfg["liquidity"]["max_orderbook_depth_fraction"],
        max_acceptable_slippage=risk_cfg["liquidity"]["max_acceptable_slippage"],
        circuit_breaker_lookback_trades=risk_cfg["circuit_breaker"]["lookback_trades"],
        risk_of_ruin_halt_threshold=risk_cfg["circuit_breaker"]["risk_of_ruin_halt_threshold"],
        max_drawdown_hard_stop=risk_cfg["circuit_breaker"]["max_drawdown_hard_stop"],
        starting_capital=config["bankroll"]["starting_capital_usd"],
    )

    result = simulate(candidates, params)
    report = build_report(result, params.starting_capital)
    click.echo(format_report_text(report))

    if sensitivity and candidates:
        click.echo("\n=== Sensitivity Analysis ===")
        points = run_sensitivity_grid(
            candidates, params,
            edge_thresholds=[0.01, 0.03, 0.05, 0.08],
            kelly_fraction_caps=[0.1, 0.25, 0.5],
            fee_multiplier_scales=[1.0, 1.5, 2.0],
        )
        click.echo(format_sensitivity_table(points))

    store.close()


if __name__ == "__main__":
    main()

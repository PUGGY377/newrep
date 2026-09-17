"""Realistic fill simulation for the backtester.

Uses the same order-book-walking logic as the live risk/liquidity module
(risk/liquidity.estimate_fill) so the backtest and live sizing agree on
what "realistic slippage" means -- there is exactly one slippage model in
this codebase, not a lenient one for backtesting and a strict one for
live trading.

Simplification, documented rather than hidden: fills are simulated
against a single order-book snapshot taken at decision time, fully or
partially filled immediately, with no requoting across multiple
snapshots and no modeling of other participants reacting to our own
order. This is standard for a first-pass backtest and is exactly why the
execution layer treats live paper/real fills as the ground truth to
eventually validate this simulation against -- see execution/paper.py.
"""

from __future__ import annotations

from dataclasses import dataclass

from kalshi_quant.data.models import OrderbookSnapshot, Side
from kalshi_quant.risk.liquidity import estimate_fill


@dataclass(frozen=True)
class SimulatedFill:
    contracts_filled: int
    average_price_probability: float
    total_cost_dollars: float


def simulate_fill(orderbook: OrderbookSnapshot, side: Side, requested_contracts: int) -> SimulatedFill:
    if requested_contracts <= 0:
        return SimulatedFill(0, 0.0, 0.0)
    fill = estimate_fill(orderbook, side, requested_contracts)
    return SimulatedFill(
        contracts_filled=fill.contracts_fillable,
        average_price_probability=fill.average_price_probability,
        total_cost_dollars=fill.contracts_fillable * fill.average_price_probability,
    )

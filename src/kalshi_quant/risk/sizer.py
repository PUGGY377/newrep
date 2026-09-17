"""Top-level position sizer: combines Kelly+shrinkage, variance budgeting,
hard fractional ceilings, and order-book liquidity into one final stake.

Final stake = min(
    kelly_stake,               # risk/kelly.py: edge-driven, confidence-shrunk
    variance_budget_stake,     # risk/portfolio.py: correlation-aware
    hard_ceiling_stake,        # risk/portfolio.py: fixed backstop fractions
    liquidity_stake,           # risk/liquidity.py: order-book-depth-limited
)

Every one of these is a genuine constraint, not a preference -- taking
the min (rather than e.g. an average) is deliberate: any single one of
them being restrictive is a valid reason to trade smaller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from kalshi_quant.data.models import OrderbookSnapshot
from kalshi_quant.risk.kelly import KellyResult, compute_kelly_fraction
from kalshi_quant.risk.liquidity import max_contracts_within_slippage
from kalshi_quant.risk.portfolio import (
    BankrollState,
    Portfolio,
    PortfolioConstraints,
    hard_ceiling_stake_cap,
    variance_budget_stake_cap,
)
from kalshi_quant.signal.edge import Edge


@dataclass(frozen=True)
class PositionSize:
    market_ticker: str
    contracts: int
    stake_dollars: float
    binding_constraint: str  # which of the four caps was the tightest, for audit
    kelly: KellyResult


def size_position(
    edge: Edge,
    variance: float,
    orderbook: OrderbookSnapshot,
    bankroll: BankrollState,
    portfolio: Portfolio,
    cluster_id: str,
    category: str,
    kelly_shrinkage_k: float,
    max_kelly_fraction: float,
    portfolio_constraints: PortfolioConstraints,
    max_orderbook_depth_fraction: float,
    max_acceptable_slippage: float,
) -> PositionSize:
    kelly = compute_kelly_fraction(edge, variance, kelly_shrinkage_k, max_kelly_fraction)
    if kelly.capped_fraction <= 0.0:
        return PositionSize(
            market_ticker=edge.market_ticker, contracts=0, stake_dollars=0.0,
            binding_constraint="no_edge_after_shrinkage", kelly=kelly,
        )

    kelly_stake = kelly.capped_fraction * bankroll.current_bankroll

    variance_stake = variance_budget_stake_cap(
        portfolio, bankroll, portfolio_constraints, cluster_id,
        edge.model_probability, edge.execution_price_probability,
    )
    ceiling_stake = hard_ceiling_stake_cap(portfolio, bankroll, portfolio_constraints, cluster_id, category)

    max_liquidity_contracts = max_contracts_within_slippage(
        orderbook, edge.side, max_acceptable_slippage, max_orderbook_depth_fraction,
    )
    liquidity_stake = max_liquidity_contracts * edge.execution_price_probability

    candidates = {
        "kelly": kelly_stake,
        "variance_budget": variance_stake,
        "hard_ceiling": ceiling_stake,
        "liquidity": liquidity_stake,
    }
    binding = min(candidates, key=lambda k: candidates[k])
    final_stake = max(candidates[binding], 0.0)

    if edge.execution_price_probability <= 0:
        contracts = 0
    else:
        contracts = math.floor(final_stake / edge.execution_price_probability)
    final_stake_rounded = contracts * edge.execution_price_probability

    return PositionSize(
        market_ticker=edge.market_ticker, contracts=contracts, stake_dollars=final_stake_rounded,
        binding_constraint=binding, kelly=kelly,
    )

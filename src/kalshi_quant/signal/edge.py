"""Edge computation: fee- and slippage-adjusted expected value.

For a binary contract paying $1 on YES:
  raw edge (buying YES at price p)  = p_hat - p
  raw edge (buying NO at price q)   = (1 - p_hat) - q

`p`/`q` here must be the price you would ACTUALLY pay -- the best ask,
not the mid -- because the mid price is not executable. Using the mid
overstates edge by half the spread on every single trade, which is
exactly the kind of error that looks fine in a naive backtest and loses
money live.

Net edge subtracts both the fee (signal/fees.py) and an explicit
slippage estimate (from risk/liquidity.py, which walks the order book)
before any threshold is applied.
"""

from __future__ import annotations

from dataclasses import dataclass

from kalshi_quant.data.models import OrderbookSnapshot, Side
from kalshi_quant.forecasting.base import ProbabilityEstimate
from kalshi_quant.signal.fees import FeeSchedule


@dataclass(frozen=True)
class Edge:
    market_ticker: str
    side: Side  # which side we'd buy to capture the edge
    model_probability: float
    model_confidence: float
    execution_price_probability: float  # best ask actually payable, in [0,1]
    expected_slippage_probability: float  # additional adverse price impact, in probability units
    fee_per_contract_dollars: float
    raw_edge: float  # model_probability - naive market-implied probability (mid), informational only
    net_edge_per_contract: float  # per $1-notional contract, after price, slippage, and fees

    @property
    def is_positive(self) -> bool:
        return self.net_edge_per_contract > 0.0


class NoLiquidityError(RuntimeError):
    """Raised when the relevant side of the book has no resting orders --
    there is no price to trade at, so no edge can be computed."""


def market_implied_probability(orderbook: OrderbookSnapshot, side: Side) -> float:
    """Mid-price-implied probability of the given side settling YES-true
    (for NO, this is 1 - yes_mid). Informational / for `raw_edge` only --
    never use this as the assumed execution price."""
    yes_mid = orderbook.implied_probability
    if yes_mid is None:
        raise NoLiquidityError(f"No two-sided market for {orderbook.market_ticker}")
    return yes_mid if side == Side.YES else 1.0 - yes_mid


def executable_price(orderbook: OrderbookSnapshot, side: Side) -> float:
    """The best price at which `contracts` of `side` could actually be
    bought right now, as a probability in [0, 1]. Buying YES executes
    against yes_asks; buying NO executes against no_asks."""
    book = orderbook.yes_asks if side == Side.YES else orderbook.no_asks
    if not book:
        raise NoLiquidityError(f"No resting {side.value} asks for {orderbook.market_ticker}")
    return book[0].price_cents / 100.0


def compute_edge(
    estimate: ProbabilityEstimate,
    orderbook: OrderbookSnapshot,
    side: Side,
    series_ticker: str,
    fee_schedule: FeeSchedule,
    expected_slippage_probability: float = 0.0,
    contracts_for_fee_estimate: int = 1,
) -> Edge:
    if estimate.market_ticker != orderbook.market_ticker:
        raise ValueError("estimate and orderbook are for different markets")

    model_prob_for_side = estimate.probability if side == Side.YES else 1.0 - estimate.probability
    exec_price = executable_price(orderbook, side)
    fee_per_contract = fee_schedule.fee_dollars(series_ticker, contracts_for_fee_estimate, exec_price)
    fee_per_contract_normalized = fee_per_contract / max(contracts_for_fee_estimate, 1)

    raw_edge = model_prob_for_side - market_implied_probability(orderbook, side)
    net_edge = (
        model_prob_for_side
        - exec_price
        - expected_slippage_probability
        - fee_per_contract_normalized
    )

    return Edge(
        market_ticker=estimate.market_ticker,
        side=side,
        model_probability=model_prob_for_side,
        model_confidence=estimate.confidence,
        execution_price_probability=exec_price,
        expected_slippage_probability=expected_slippage_probability,
        fee_per_contract_dollars=fee_per_contract_normalized,
        raw_edge=raw_edge,
        net_edge_per_contract=net_edge,
    )


def passes_signal_filter(edge: Edge, min_edge_after_fees: float, min_confidence: float) -> bool:
    """Both gates must pass -- a large edge from a low-confidence forecaster
    is exactly the case the spec calls out as dangerous to trade on."""
    return edge.net_edge_per_contract >= min_edge_after_fees and edge.model_confidence >= min_confidence

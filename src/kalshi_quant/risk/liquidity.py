"""Order-book-aware slippage estimation and size capping.

Walks the resting order book depth to answer two questions:
  1. What would the average execution price be for buying N contracts?
  2. Given a max acceptable slippage, what's the largest N we should size
     for -- so the position sizer never proposes a stake larger than the
     book can actually absorb without eroding the edge that motivated the
     trade in the first place.
"""

from __future__ import annotations

from dataclasses import dataclass

from kalshi_quant.data.models import OrderbookLevel, OrderbookSnapshot, Side


@dataclass(frozen=True)
class FillEstimate:
    contracts_fillable: int
    average_price_probability: float
    slippage_vs_best_price: float  # average_price - best_price, in probability units


def _book_for(orderbook: OrderbookSnapshot, side: Side) -> tuple[OrderbookLevel, ...]:
    return orderbook.yes_asks if side == Side.YES else orderbook.no_asks


def estimate_fill(orderbook: OrderbookSnapshot, side: Side, contracts: int) -> FillEstimate:
    """Simulate walking the book to fill `contracts`, worst realistic case
    (no assumption of hidden/iceberg liquidity beyond what's displayed)."""
    book = _book_for(orderbook, side)
    if not book:
        return FillEstimate(0, 0.0, 0.0)

    best_price = book[0].price_cents / 100.0
    remaining = contracts
    total_cost = 0.0
    filled = 0
    for level in book:
        if remaining <= 0:
            break
        take = min(remaining, level.contracts)
        total_cost += take * (level.price_cents / 100.0)
        filled += take
        remaining -= take

    if filled == 0:
        return FillEstimate(0, 0.0, 0.0)
    avg_price = total_cost / filled
    return FillEstimate(
        contracts_fillable=filled, average_price_probability=avg_price,
        slippage_vs_best_price=avg_price - best_price,
    )


def max_contracts_within_slippage(
    orderbook: OrderbookSnapshot, side: Side, max_acceptable_slippage: float,
    max_orderbook_depth_fraction: float,
) -> int:
    """Largest contract count such that (a) average execution price does not
    exceed best price + max_acceptable_slippage, and (b) we never take more
    than `max_orderbook_depth_fraction` of the total visible depth on that
    side (a backstop against being the entire market for a thin book, which
    would also move the price on any future exit/hedge)."""
    book = _book_for(orderbook, side)
    if not book:
        return 0

    total_depth = sum(level.contracts for level in book)
    depth_cap = int(total_depth * max_orderbook_depth_fraction)
    if depth_cap <= 0:
        return 0

    best_price = book[0].price_cents / 100.0
    remaining = depth_cap
    total_cost = 0.0
    filled = 0
    for level in book:
        if remaining <= 0:
            break
        take = min(remaining, level.contracts)
        candidate_cost = total_cost + take * (level.price_cents / 100.0)
        candidate_filled = filled + take
        candidate_avg = candidate_cost / candidate_filled
        if candidate_avg - best_price > max_acceptable_slippage:
            # Binary-search within this level for the largest `take` that
            # keeps us under the slippage budget, rather than stopping
            # abruptly at the level boundary.
            lo, hi = 0, take
            while lo < hi:
                mid = (lo + hi + 1) // 2
                trial_cost = total_cost + mid * (level.price_cents / 100.0)
                trial_filled = filled + mid
                if trial_filled > 0 and (trial_cost / trial_filled - best_price) <= max_acceptable_slippage:
                    lo = mid
                else:
                    hi = mid - 1
            filled += lo
            break
        total_cost, filled, remaining = candidate_cost, candidate_filled, remaining - take

    return filled

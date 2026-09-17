"""Fractional Kelly position sizing with confidence-based shrinkage.

Derivation of the base Kelly fraction for a Kalshi-style binary contract:
staking fraction f of bankroll to buy at price `p` (in probability units,
i.e. cents/100) that pays $1 if YES: wealth multiplies by (1 + f*b) with
probability p_hat and by (1 - f) with probability (1 - p_hat), where
b = (1-p)/p is the payout odds. Maximizing E[log(wealth)] gives:

    f* = (b*p_hat - (1-p_hat)) / b  =  (p_hat - p) / (1 - p)

i.e. base Kelly fraction = net edge / (1 - effective price), where
"effective price" folds in fees and expected slippage (see signal/edge.py)
so the formula is optimizing against what you actually pay, not the
theoretical mid.

This is THE mathematically-driven core of "how much to bet" -- nothing
here is a hand-picked percentage. What follows Kelly (shrinkage, hard
caps) exists because full Kelly against a model whose calibration is
itself uncertain is known to be dangerous: Kelly assumes p_hat is exactly
right, and errors in p_hat are punished asymmetrically (overbetting hurts
more than underbetting by the same amount). Shrinkage makes the stake
size sensitive to how much we actually trust p_hat.
"""

from __future__ import annotations

from dataclasses import dataclass

from kalshi_quant.signal.edge import Edge


@dataclass(frozen=True)
class KellyResult:
    market_ticker: str
    base_kelly_fraction: float  # f*, can be negative if no edge (caller should clip)
    shrinkage_factor: float  # in (0, 1]
    shrunk_fraction: float  # base_kelly_fraction * shrinkage_factor, before hard cap
    capped_fraction: float  # final fraction of bankroll to stake, in [0, max_kelly_fraction]
    effective_price: float  # execution price + slippage + fee, in probability units


def effective_price(edge: Edge) -> float:
    return (
        edge.execution_price_probability
        + edge.expected_slippage_probability
        + edge.fee_per_contract_dollars
    )


def base_kelly_fraction(edge: Edge) -> float:
    p_eff = effective_price(edge)
    if p_eff >= 1.0:
        return 0.0
    return edge.net_edge_per_contract / (1.0 - p_eff)


def confidence_shrinkage(variance: float, shrinkage_k: float) -> float:
    """shrink = 1 / (1 + k * variance). variance=0 (perfect confidence) ->
    shrink=1 (no reduction). Higher variance or higher k -> more shrinkage.
    `shrinkage_k` is a config parameter, tuned via backtest sensitivity
    analysis (config.risk.kelly.shrinkage_k) -- it is not re-derived per
    trade because it represents our general trust in the forecaster's
    self-reported uncertainty, which is a property of the model, not of
    any single prediction."""
    if shrinkage_k < 0:
        raise ValueError("shrinkage_k must be >= 0")
    return 1.0 / (1.0 + shrinkage_k * max(variance, 0.0))


def compute_kelly_fraction(
    edge: Edge, variance: float, shrinkage_k: float, max_kelly_fraction: float
) -> KellyResult:
    if not edge.is_positive:
        return KellyResult(
            market_ticker=edge.market_ticker, base_kelly_fraction=0.0, shrinkage_factor=1.0,
            shrunk_fraction=0.0, capped_fraction=0.0, effective_price=effective_price(edge),
        )

    f_base = base_kelly_fraction(edge)
    shrink = confidence_shrinkage(variance, shrinkage_k)
    f_shrunk = max(f_base, 0.0) * shrink
    f_capped = min(f_shrunk, max_kelly_fraction)

    return KellyResult(
        market_ticker=edge.market_ticker, base_kelly_fraction=f_base, shrinkage_factor=shrink,
        shrunk_fraction=f_shrunk, capped_fraction=f_capped, effective_price=effective_price(edge),
    )

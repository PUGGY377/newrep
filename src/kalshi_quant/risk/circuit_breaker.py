"""Drawdown circuit breaker, derived from a risk-of-ruin estimate rather
than an arbitrary drawdown percentage.

Approach: model the log-TOTAL-BANKROLL process under the strategy's
recent trade-by-trade results as a Brownian motion with drift mu (mean
log return of bankroll per trade) and variance sigma^2, both estimated
empirically from the last `lookback_trades` settled trades. Standard
first-passage result for Brownian motion with positive drift: the
probability of ever falling by `a` log-units below the current level is

    P(ruin) = exp(-2 * mu * a / sigma^2)     for mu > 0

where `a = -ln(1 - drawdown_fraction)` converts a drawdown fraction into
log-space. If mu <= 0 (the strategy has no measured edge over the
lookback window), ruin is treated as certain (P=1).

CRITICAL: the log-return per trade must be computed relative to TOTAL
BANKROLL before/after the trade, i.e. ln(bankroll_after / bankroll_before)
-- NOT relative to the stake risked on that single trade. A fractionally-
sized bet (Kelly or otherwise) that loses entirely reduces bankroll by
only that fraction, not to zero; computing the "wealth multiple" as
1 + pnl/stake instead conflates "lost the whole stake" with "lost the
whole bankroll" and produces wildly overstated ruin probabilities for any
strategy that (correctly) never bets 100% of bankroll on one trade.

This makes the breaker's trigger point a function of the strategy's
*actual recent performance*, not a fixed number invented up front: a
strategy performing well (positive, low-variance drift) tolerates more
drawdown before halting; one whose edge is degrading trips sooner. The
config's `max_drawdown_hard_stop` remains as a hard backstop in case the
mu/sigma estimate itself is unreliable (e.g. too few trades).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Below this many settled trades, the empirical mu/sigma estimate is too
# noisy to trust at all -- estimate_risk_of_ruin reports maximal caution
# (P=1) rather than a falsely precise number. should_halt_new_positions
# deliberately does NOT act on that maximal-caution reading (see below);
# it would otherwise trip permanently on trade #1 of any fresh run, since
# there is no way to ever accumulate enough trades to escape it once
# halted. Only the hard drawdown backstop applies during this bootstrap
# window.
MIN_TRADES_FOR_RUIN_ESTIMATE = 5


@dataclass(frozen=True)
class RuinEstimate:
    mu: float  # mean log-return of BANKROLL per trade over the lookback window
    sigma_sq: float  # variance of log-return of bankroll per trade
    n_trades: int
    probability_of_ruin_at_current_drawdown: float
    probability_of_ruin_at_hard_stop: float


def _log_returns_from_bankroll(bankroll_before: list[float], bankroll_after: list[float]) -> list[float]:
    if len(bankroll_before) != len(bankroll_after):
        raise ValueError("bankroll_before and bankroll_after must be the same length")
    log_returns = []
    for before, after in zip(bankroll_before, bankroll_after):
        if before <= 0:
            continue
        wealth_multiple = after / before
        if wealth_multiple <= 0:
            wealth_multiple = 1e-6  # total loss of bankroll; avoid log(0)
        log_returns.append(math.log(wealth_multiple))
    return log_returns


def estimate_risk_of_ruin(
    bankroll_before: list[float], bankroll_after: list[float],
    current_drawdown_fraction: float, hard_stop_drawdown_fraction: float,
) -> RuinEstimate:
    log_returns = _log_returns_from_bankroll(bankroll_before, bankroll_after)
    n = len(log_returns)
    if n < MIN_TRADES_FOR_RUIN_ESTIMATE:
        # Not enough history to trust an empirical mu/sigma -- report
        # maximal caution rather than a falsely precise number.
        return RuinEstimate(
            mu=0.0, sigma_sq=0.0, n_trades=n,
            probability_of_ruin_at_current_drawdown=1.0,
            probability_of_ruin_at_hard_stop=1.0,
        )

    mu = sum(log_returns) / n
    sigma_sq = sum((r - mu) ** 2 for r in log_returns) / max(n - 1, 1)
    sigma_sq = max(sigma_sq, 1e-9)

    def ruin_prob(drawdown_fraction: float) -> float:
        if drawdown_fraction <= 0:
            return 0.0
        if drawdown_fraction >= 1:
            return 1.0
        if mu <= 0:
            return 1.0
        a = -math.log(1 - drawdown_fraction)
        return math.exp(-2 * mu * a / sigma_sq)

    return RuinEstimate(
        mu=mu, sigma_sq=sigma_sq, n_trades=n,
        probability_of_ruin_at_current_drawdown=ruin_prob(current_drawdown_fraction),
        probability_of_ruin_at_hard_stop=ruin_prob(hard_stop_drawdown_fraction),
    )


def should_halt_new_positions(
    bankroll_before: list[float], bankroll_after: list[float],
    current_drawdown_fraction: float, risk_of_ruin_halt_threshold: float,
    max_drawdown_hard_stop: float,
) -> tuple[bool, str]:
    """Returns (halt, reason). Two independent triggers, either one halts:
    the computed risk-of-ruin exceeding its threshold, or the absolute
    hard-stop drawdown backstop being breached regardless of the model."""
    if current_drawdown_fraction >= max_drawdown_hard_stop:
        return True, (
            f"Hard-stop drawdown breached: {current_drawdown_fraction:.1%} >= "
            f"{max_drawdown_hard_stop:.1%}"
        )

    if len(bankroll_before) < MIN_TRADES_FOR_RUIN_ESTIMATE:
        return False, "ok (insufficient trade history for risk-of-ruin estimate; hard stop still active)"

    estimate = estimate_risk_of_ruin(
        bankroll_before, bankroll_after, current_drawdown_fraction, max_drawdown_hard_stop,
    )
    if estimate.probability_of_ruin_at_current_drawdown >= risk_of_ruin_halt_threshold:
        return True, (
            f"Computed risk-of-ruin {estimate.probability_of_ruin_at_current_drawdown:.1%} "
            f"(mu={estimate.mu:.5f}, sigma^2={estimate.sigma_sq:.5f}, n={estimate.n_trades}) "
            f">= threshold {risk_of_ruin_halt_threshold:.1%}"
        )
    return False, "ok"

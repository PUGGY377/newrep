"""Portfolio state and correlation-aware variance budgeting.

Sizing a new position is not just "Kelly fraction times bankroll" -- it
also has to respect how much portfolio-level variance is already
committed to correlated markets. The approach here:

1. Each position's own dollar-payoff variance has a closed form for a
   binary contract: staking `stake` dollars at price `p` (contracts =
   stake/p) to win $1/contract with probability p_hat has payoff
   variance = p_hat*(1-p_hat) * contracts^2 = p_hat*(1-p_hat)*(stake/p)^2.
   (Derivation: profit if YES = contracts*(1-p); loss if NO = stake;
   the swing between outcomes is contracts*(1-p) + stake = contracts, so
   Var = p_hat*(1-p_hat) * contracts^2.)

2. Within a correlation cluster (signal/correlation.py), positions are
   treated as PERFECTLY correlated (rho=1) -- the conservative
   worst case appropriate for markets tied to the same underlying event.
   Combined cluster variance = (sum of each position's payoff std-dev)^2,
   not the sum of variances.

3. Across clusters, positions are treated as independent (rho=0) by
   default -- different underlying events -- so total portfolio variance
   is the sum of per-cluster variances. This is a simplification (real
   cross-event correlation is rarely exactly zero); it is a deliberate,
   documented limitation, not an oversight. Extending this to a full
   covariance matrix estimated from historical co-movement is future
   work once there's enough live data to estimate it reliably.

4. A new candidate position is sized so total portfolio variance stays
   under `target_portfolio_variance_frac * bankroll^2`, in addition to
   (not instead of) the Kelly-derived stake and the hard fractional
   ceilings in config, all of which are combined with `min()`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from kalshi_quant.data.models import Side


@dataclass
class Position:
    market_ticker: str
    cluster_id: str
    category: str
    side: Side
    contracts: int
    entry_price_probability: float
    model_probability_at_entry: float
    stake_dollars: float

    @property
    def payoff_std_dev(self) -> float:
        variance = self.model_probability_at_entry * (1 - self.model_probability_at_entry) * (
            self.contracts ** 2
        )
        return math.sqrt(max(variance, 0.0))


@dataclass
class BankrollState:
    starting_capital: float
    cash: float
    realized_pnl: float = 0.0

    @property
    def current_bankroll(self) -> float:
        """Cash plus realized P&L -- the base against which all fractional
        sizing (Kelly and hard caps) is computed. Open positions'
        unrealized mark-to-market is intentionally excluded from the
        sizing base to avoid a feedback loop where paper gains on open,
        unsettled positions inflate the stake on new, unrelated bets."""
        return self.starting_capital + self.realized_pnl


@dataclass
class Portfolio:
    positions: list[Position] = field(default_factory=list)

    def positions_in_cluster(self, cluster_id: str) -> list[Position]:
        return [p for p in self.positions if p.cluster_id == cluster_id]

    def positions_in_category(self, category: str) -> list[Position]:
        return [p for p in self.positions if p.category == category]

    def cluster_variance(self, cluster_id: str) -> float:
        sigma_sum = sum(p.payoff_std_dev for p in self.positions_in_cluster(cluster_id))
        return sigma_sum ** 2

    def total_variance(self) -> float:
        cluster_ids = {p.cluster_id for p in self.positions}
        return sum(self.cluster_variance(cid) for cid in cluster_ids)

    def total_exposure_dollars(self) -> float:
        return sum(p.stake_dollars for p in self.positions)

    def category_exposure_dollars(self, category: str) -> float:
        return sum(p.stake_dollars for p in self.positions_in_category(category))

    def cluster_exposure_dollars(self, cluster_id: str) -> float:
        return sum(p.stake_dollars for p in self.positions_in_cluster(cluster_id))


@dataclass(frozen=True)
class PortfolioConstraints:
    target_portfolio_variance_frac: float
    max_fraction_per_market: float
    max_fraction_per_category: float
    max_fraction_total_exposure: float


def variance_budget_stake_cap(
    portfolio: Portfolio, bankroll: BankrollState, constraints: PortfolioConstraints,
    cluster_id: str, model_probability: float, price_probability: float,
) -> float:
    """Max stake (dollars) for a new position in `cluster_id` such that
    total portfolio variance stays within budget. Returns 0.0 if the
    budget is already exhausted or `price_probability` is degenerate."""
    if price_probability <= 0.0:
        return 0.0

    total_budget = constraints.target_portfolio_variance_frac * bankroll.current_bankroll ** 2
    current_total_variance = portfolio.total_variance()
    current_cluster_variance = portfolio.cluster_variance(cluster_id)
    budget_available_for_cluster = total_budget - (current_total_variance - current_cluster_variance)
    if budget_available_for_cluster <= 0:
        return 0.0

    sigma_cluster_current = math.sqrt(current_cluster_variance)
    max_total_sigma = math.sqrt(budget_available_for_cluster)
    sigma_new_budget = max_total_sigma - sigma_cluster_current
    if sigma_new_budget <= 0:
        return 0.0

    per_contract_sigma = math.sqrt(max(model_probability * (1 - model_probability), 1e-9))
    max_contracts = sigma_new_budget / per_contract_sigma
    return max_contracts * price_probability


def hard_ceiling_stake_cap(
    portfolio: Portfolio, bankroll: BankrollState, constraints: PortfolioConstraints,
    cluster_id: str, category: str,
) -> float:
    """Backstop caps expressed as fractions of current bankroll -- these
    clip the mathematically-derived stake, they never drive it directly."""
    bankroll_amount = bankroll.current_bankroll
    market_cap = constraints.max_fraction_per_market * bankroll_amount
    category_remaining = max(
        0.0, constraints.max_fraction_per_category * bankroll_amount
        - portfolio.category_exposure_dollars(category)
    )
    total_remaining = max(
        0.0, constraints.max_fraction_total_exposure * bankroll_amount
        - portfolio.total_exposure_dollars()
    )
    return min(market_cap, category_remaining, total_remaining)

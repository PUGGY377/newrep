import pytest

from kalshi_quant.data.models import Side
from kalshi_quant.risk.portfolio import (
    BankrollState,
    Portfolio,
    PortfolioConstraints,
    Position,
    hard_ceiling_stake_cap,
    variance_budget_stake_cap,
)


def make_bankroll(amount=500.0) -> BankrollState:
    return BankrollState(starting_capital=amount, cash=amount)


def make_constraints(**overrides) -> PortfolioConstraints:
    defaults = dict(
        target_portfolio_variance_frac=0.04, max_fraction_per_market=0.25,
        max_fraction_per_category=0.60, max_fraction_total_exposure=0.85,
    )
    defaults.update(overrides)
    return PortfolioConstraints(**defaults)


def test_position_payoff_std_dev_matches_binary_formula():
    pos = Position(
        market_ticker="A", cluster_id="c1", category="sports", side=Side.YES,
        contracts=100, entry_price_probability=0.5, model_probability_at_entry=0.6,
        stake_dollars=50.0,
    )
    import math
    expected = math.sqrt(0.6 * 0.4 * 100 ** 2)
    assert pos.payoff_std_dev == pytest.approx(expected)


def test_variance_budget_shrinks_as_cluster_fills_up():
    bankroll = make_bankroll(500.0)
    constraints = make_constraints()
    portfolio = Portfolio()

    first_cap = variance_budget_stake_cap(
        portfolio, bankroll, constraints, cluster_id="c1",
        model_probability=0.6, price_probability=0.5,
    )
    assert first_cap > 0

    portfolio.positions.append(
        Position(
            market_ticker="A", cluster_id="c1", category="sports", side=Side.YES,
            contracts=first_cap / 0.5, entry_price_probability=0.5,
            model_probability_at_entry=0.6, stake_dollars=first_cap,
        )
    )
    second_cap = variance_budget_stake_cap(
        portfolio, bankroll, constraints, cluster_id="c1",
        model_probability=0.6, price_probability=0.5,
    )
    assert second_cap == pytest.approx(0.0, abs=1e-6)


def test_variance_budget_independent_across_clusters():
    bankroll = make_bankroll(500.0)
    constraints = make_constraints()
    portfolio = Portfolio()
    portfolio.positions.append(
        Position(
            market_ticker="A", cluster_id="c1", category="sports", side=Side.YES,
            contracts=200, entry_price_probability=0.5, model_probability_at_entry=0.6,
            stake_dollars=100.0,
        )
    )
    cap_other_cluster = variance_budget_stake_cap(
        portfolio, bankroll, constraints, cluster_id="c2",
        model_probability=0.6, price_probability=0.5,
    )
    assert cap_other_cluster > 0


def test_hard_ceiling_respects_category_and_total_caps():
    bankroll = make_bankroll(500.0)
    constraints = make_constraints(
        max_fraction_per_market=0.25, max_fraction_per_category=0.30, max_fraction_total_exposure=0.85,
    )
    portfolio = Portfolio()
    portfolio.positions.append(
        Position(
            market_ticker="A", cluster_id="c1", category="sports", side=Side.YES,
            contracts=100, entry_price_probability=1.0, model_probability_at_entry=0.6,
            stake_dollars=100.0,  # already 20% of bankroll in "sports" category
        )
    )
    cap = hard_ceiling_stake_cap(portfolio, bankroll, constraints, cluster_id="c2", category="sports")
    # category cap remaining = 0.30*500 - 100 = 50; market cap = 0.25*500=125; total remaining = 0.85*500-100=325
    assert cap == pytest.approx(50.0)

import pytest

from kalshi_quant.data.models import Side
from kalshi_quant.risk.portfolio import BankrollState, Portfolio, PortfolioConstraints
from kalshi_quant.risk.sizer import size_position
from kalshi_quant.signal.edge import Edge
from tests.conftest import make_orderbook


def make_edge(model_probability=0.75, execution_price=0.55) -> Edge:
    net_edge = model_probability - execution_price
    return Edge(
        market_ticker="KXTEST-A", side=Side.YES, model_probability=model_probability,
        model_confidence=0.9, execution_price_probability=execution_price,
        expected_slippage_probability=0.0, fee_per_contract_dollars=0.0,
        raw_edge=net_edge, net_edge_per_contract=net_edge,
    )


def default_constraints() -> PortfolioConstraints:
    return PortfolioConstraints(
        target_portfolio_variance_frac=0.04, max_fraction_per_market=0.25,
        max_fraction_per_category=0.60, max_fraction_total_exposure=0.85,
    )


def test_size_position_returns_zero_for_negative_edge():
    ob = make_orderbook(yes_asks=[(55, 1000)])
    edge = make_edge(model_probability=0.5, execution_price=0.55)  # negative
    result = size_position(
        edge, variance=0.02, orderbook=ob, bankroll=BankrollState(500, 500),
        portfolio=Portfolio(), cluster_id="c1", category="sports",
        kelly_shrinkage_k=8.0, max_kelly_fraction=0.5, portfolio_constraints=default_constraints(),
        max_orderbook_depth_fraction=0.25, max_acceptable_slippage=0.02,
    )
    assert result.contracts == 0
    assert result.stake_dollars == 0.0


def test_size_position_positive_edge_produces_stake_within_bankroll():
    ob = make_orderbook(yes_asks=[(55, 1000)])
    edge = make_edge(model_probability=0.75, execution_price=0.55)
    result = size_position(
        edge, variance=0.02, orderbook=ob, bankroll=BankrollState(500, 500),
        portfolio=Portfolio(), cluster_id="c1", category="sports",
        kelly_shrinkage_k=8.0, max_kelly_fraction=0.5, portfolio_constraints=default_constraints(),
        max_orderbook_depth_fraction=0.25, max_acceptable_slippage=0.02,
    )
    assert result.contracts > 0
    assert result.stake_dollars <= 500 * 0.25 + 1e-6  # never exceeds per-market hard ceiling
    assert result.stake_dollars <= 500  # never exceeds bankroll


def test_size_position_liquidity_binds_on_thin_book():
    ob = make_orderbook(yes_asks=[(55, 3)])  # only 3 contracts available
    edge = make_edge(model_probability=0.9, execution_price=0.55)  # huge edge, would want big stake
    result = size_position(
        edge, variance=0.001, orderbook=ob, bankroll=BankrollState(500, 500),
        portfolio=Portfolio(), cluster_id="c1", category="sports",
        kelly_shrinkage_k=1.0, max_kelly_fraction=1.0, portfolio_constraints=default_constraints(),
        max_orderbook_depth_fraction=1.0, max_acceptable_slippage=1.0,
    )
    assert result.binding_constraint == "liquidity"
    assert result.contracts <= 3


def test_size_position_records_binding_constraint():
    ob = make_orderbook(yes_asks=[(55, 1000)])
    edge = make_edge(model_probability=0.75, execution_price=0.55)
    result = size_position(
        edge, variance=0.02, orderbook=ob, bankroll=BankrollState(500, 500),
        portfolio=Portfolio(), cluster_id="c1", category="sports",
        kelly_shrinkage_k=8.0, max_kelly_fraction=0.5, portfolio_constraints=default_constraints(),
        max_orderbook_depth_fraction=0.25, max_acceptable_slippage=0.02,
    )
    assert result.binding_constraint in {"kelly", "variance_budget", "hard_ceiling", "liquidity"}

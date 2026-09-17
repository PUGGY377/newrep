import pytest

from kalshi_quant.data.models import Side
from kalshi_quant.risk.kelly import (
    base_kelly_fraction,
    compute_kelly_fraction,
    confidence_shrinkage,
    effective_price,
)
from kalshi_quant.signal.edge import Edge


def make_edge(model_probability=0.7, execution_price=0.6, slippage=0.0, fee=0.0) -> Edge:
    net_edge = model_probability - execution_price - slippage - fee
    return Edge(
        market_ticker="KXTEST-A", side=Side.YES, model_probability=model_probability,
        model_confidence=0.9, execution_price_probability=execution_price,
        expected_slippage_probability=slippage, fee_per_contract_dollars=fee,
        raw_edge=net_edge, net_edge_per_contract=net_edge,
    )


def test_base_kelly_matches_closed_form_binary_formula():
    # f* = (p_hat - price) / (1 - price) with no fees/slippage
    edge = make_edge(model_probability=0.7, execution_price=0.6)
    f = base_kelly_fraction(edge)
    assert f == pytest.approx((0.7 - 0.6) / (1 - 0.6))


def test_base_kelly_zero_at_zero_edge():
    edge = make_edge(model_probability=0.6, execution_price=0.6)
    assert base_kelly_fraction(edge) == pytest.approx(0.0)


def test_confidence_shrinkage_bounds():
    assert confidence_shrinkage(0.0, shrinkage_k=8.0) == pytest.approx(1.0)
    assert 0.0 < confidence_shrinkage(0.25, shrinkage_k=8.0) < 1.0
    with pytest.raises(ValueError):
        confidence_shrinkage(0.1, shrinkage_k=-1.0)


def test_compute_kelly_fraction_applies_shrinkage_and_cap():
    edge = make_edge(model_probability=0.9, execution_price=0.5)  # big raw edge
    result = compute_kelly_fraction(edge, variance=0.2, shrinkage_k=8.0, max_kelly_fraction=0.5)
    assert result.base_kelly_fraction == pytest.approx((0.9 - 0.5) / 0.5)
    assert result.shrunk_fraction < result.base_kelly_fraction
    assert result.capped_fraction <= 0.5
    assert result.capped_fraction == pytest.approx(min(result.shrunk_fraction, 0.5))


def test_compute_kelly_fraction_zero_when_no_edge():
    edge = make_edge(model_probability=0.5, execution_price=0.6)  # negative edge
    result = compute_kelly_fraction(edge, variance=0.05, shrinkage_k=8.0, max_kelly_fraction=0.5)
    assert result.capped_fraction == 0.0
    assert result.base_kelly_fraction == 0.0


def test_higher_variance_shrinks_more():
    edge = make_edge(model_probability=0.8, execution_price=0.5)
    low_var = compute_kelly_fraction(edge, variance=0.01, shrinkage_k=8.0, max_kelly_fraction=1.0)
    high_var = compute_kelly_fraction(edge, variance=0.2, shrinkage_k=8.0, max_kelly_fraction=1.0)
    assert high_var.capped_fraction < low_var.capped_fraction


def test_effective_price_includes_fee_and_slippage():
    edge = make_edge(model_probability=0.7, execution_price=0.6, slippage=0.02, fee=0.01)
    assert effective_price(edge) == pytest.approx(0.63)

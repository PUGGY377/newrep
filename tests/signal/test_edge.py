import pytest

from kalshi_quant.data.models import Side
from kalshi_quant.forecasting.base import ProbabilityEstimate
from kalshi_quant.signal.edge import (
    NoLiquidityError,
    compute_edge,
    executable_price,
    market_implied_probability,
    passes_signal_filter,
)
from kalshi_quant.signal.fees import FeeSchedule
from tests.conftest import make_orderbook


def _fee_schedule():
    from datetime import datetime
    return FeeSchedule(
        version="test", effective_from=datetime(2020, 1, 1), effective_to=None,
        general_multiplier=0.0,  # zero fees to isolate edge math in most tests
    )


def test_market_implied_probability_uses_mid():
    ob = make_orderbook(yes_bids=[(50, 10)], yes_asks=[(60, 10)])
    assert market_implied_probability(ob, Side.YES) == pytest.approx(0.55)
    assert market_implied_probability(ob, Side.NO) == pytest.approx(0.45)


def test_executable_price_is_best_ask_not_mid():
    ob = make_orderbook(yes_bids=[(50, 10)], yes_asks=[(60, 10)])
    assert executable_price(ob, Side.YES) == pytest.approx(0.60)


def test_no_liquidity_raises():
    ob = make_orderbook(yes_asks=[])
    with pytest.raises(NoLiquidityError):
        executable_price(ob, Side.YES)


def test_compute_edge_positive_when_model_beats_ask():
    ob = make_orderbook(yes_bids=[(50, 10)], yes_asks=[(60, 10)])
    estimate = ProbabilityEstimate(
        market_ticker=ob.market_ticker, probability=0.75, variance=0.01,
        model_name="test", model_version="1",
    )
    edge = compute_edge(estimate, ob, Side.YES, "KXTEST", _fee_schedule())
    # net edge = 0.75 - 0.60 - 0 (slippage) - 0 (fee) = 0.15
    assert edge.net_edge_per_contract == pytest.approx(0.15)
    assert edge.is_positive


def test_compute_edge_accounts_for_fees_and_slippage():
    ob = make_orderbook(yes_bids=[(50, 10)], yes_asks=[(60, 10)])
    estimate = ProbabilityEstimate(
        market_ticker=ob.market_ticker, probability=0.62, variance=0.01,
        model_name="test", model_version="1",
    )
    from datetime import datetime
    fee_schedule = FeeSchedule(
        version="test", effective_from=datetime(2020, 1, 1), effective_to=None,
        general_multiplier=0.07,
    )
    edge = compute_edge(
        estimate, ob, Side.YES, "KXTEST", fee_schedule, expected_slippage_probability=0.01,
    )
    # raw model edge over ask = 0.62 - 0.60 = 0.02, minus 0.01 slippage minus fee > 0
    assert edge.net_edge_per_contract < 0.02 - 0.01
    assert not edge.is_positive  # fee should be enough to flip this thin edge negative


def test_passes_signal_filter_requires_both_gates():
    ob = make_orderbook(yes_bids=[(50, 10)], yes_asks=[(60, 10)])
    estimate = ProbabilityEstimate(
        market_ticker=ob.market_ticker, probability=0.75, variance=0.01,
        model_name="test", model_version="1",
    )
    edge = compute_edge(estimate, ob, Side.YES, "KXTEST", _fee_schedule())
    assert passes_signal_filter(edge, min_edge_after_fees=0.03, min_confidence=0.5)
    assert not passes_signal_filter(edge, min_edge_after_fees=0.5, min_confidence=0.5)

    low_conf_estimate = ProbabilityEstimate(
        market_ticker=ob.market_ticker, probability=0.75, variance=0.24,
        model_name="test", model_version="1",
    )
    low_conf_edge = compute_edge(low_conf_estimate, ob, Side.YES, "KXTEST", _fee_schedule())
    assert not passes_signal_filter(low_conf_edge, min_edge_after_fees=0.03, min_confidence=0.9)

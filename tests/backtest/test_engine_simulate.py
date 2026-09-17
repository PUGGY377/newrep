from datetime import datetime, timedelta

import pytest

from kalshi_quant.backtest.engine import SimParams, TradeCandidate, simulate
from kalshi_quant.data.models import Market, MarketStatus, Side
from kalshi_quant.forecasting.base import ProbabilityEstimate
from kalshi_quant.risk.portfolio import PortfolioConstraints
from kalshi_quant.signal.edge import compute_edge
from kalshi_quant.signal.fees import FeeSchedule
from tests.conftest import make_orderbook

T0 = datetime(2024, 1, 1)
ZERO_FEE_SCHEDULE = FeeSchedule(
    version="test", effective_from=datetime(2020, 1, 1), effective_to=None, general_multiplier=0.0,
)


def make_settled_market(ticker, result: Side, close_time=T0, settled_at=None) -> Market:
    return Market(
        ticker=ticker, event_ticker=f"{ticker}-EVT", title="t", subtitle="s",
        status=MarketStatus.SETTLED, open_time=close_time - timedelta(days=1), close_time=close_time,
        expiration_time=close_time, settlement_criteria="x", fetched_at=T0, result=result,
        settled_at=settled_at or close_time,
    )


def make_candidate(ticker, model_probability, ask_price, result: Side, close_time=T0,
                    cluster_id="c1", variance=0.02) -> TradeCandidate:
    market = make_settled_market(ticker, result, close_time=close_time)
    ob = make_orderbook(ticker=ticker, yes_asks=[(int(ask_price * 100), 1000)])
    estimate = ProbabilityEstimate(
        market_ticker=ticker, probability=model_probability, variance=variance,
        model_name="test", model_version="1",
    )
    edge = compute_edge(estimate, ob, Side.YES, market.event_ticker, ZERO_FEE_SCHEDULE)
    return TradeCandidate(
        market=market, decision_time=close_time - timedelta(hours=1), estimate=estimate,
        orderbook=ob, edge=edge, cluster_id=cluster_id, category="sports", series_ticker=market.event_ticker,
    )


def default_params(**overrides) -> SimParams:
    defaults = dict(
        min_edge_after_fees=0.03, min_confidence=0.5, kelly_shrinkage_k=8.0, max_kelly_fraction=0.5,
        portfolio_constraints=PortfolioConstraints(
            target_portfolio_variance_frac=0.04, max_fraction_per_market=0.25,
            max_fraction_per_category=0.60, max_fraction_total_exposure=0.85,
        ),
        max_orderbook_depth_fraction=0.5, max_acceptable_slippage=0.05,
        circuit_breaker_lookback_trades=50, risk_of_ruin_halt_threshold=0.05,
        max_drawdown_hard_stop=0.40, starting_capital=500.0,
    )
    defaults.update(overrides)
    return SimParams(**defaults)


def test_winning_trade_increases_bankroll():
    candidate = make_candidate("A", model_probability=0.75, ask_price=0.55, result=Side.YES)
    result = simulate([candidate], default_params())
    assert len(result.trades) == 1
    assert result.trades[0].pnl_dollars > 0
    assert result.final_bankroll > 500.0


def test_losing_trade_decreases_bankroll():
    candidate = make_candidate("A", model_probability=0.75, ask_price=0.55, result=Side.NO)
    result = simulate([candidate], default_params())
    assert len(result.trades) == 1
    assert result.trades[0].pnl_dollars < 0
    assert result.final_bankroll < 500.0


def test_below_threshold_edge_is_not_traded():
    candidate = make_candidate("A", model_probability=0.57, ask_price=0.55, result=Side.YES)  # edge=0.02 < 0.03
    result = simulate([candidate], default_params())
    assert len(result.trades) == 0
    assert result.final_bankroll == 500.0


def test_correlated_cluster_limits_combined_exposure_vs_independent():
    # Two markets in the SAME cluster vs two in DIFFERENT clusters, same edges.
    same_cluster = [
        make_candidate("A", 0.75, 0.55, Side.YES, cluster_id="shared"),
        make_candidate("B", 0.75, 0.55, Side.YES, cluster_id="shared"),
    ]
    diff_cluster = [
        make_candidate("C", 0.75, 0.55, Side.YES, cluster_id="c1"),
        make_candidate("D", 0.75, 0.55, Side.YES, cluster_id="c2"),
    ]
    result_same = simulate(same_cluster, default_params())
    result_diff = simulate(diff_cluster, default_params())

    total_staked_same = sum(t.contracts * t.entry_price for t in result_same.trades)
    total_staked_diff = sum(t.contracts * t.entry_price for t in result_diff.trades)
    assert total_staked_same < total_staked_diff


def test_circuit_breaker_halts_after_losing_streak():
    candidates = [
        make_candidate(f"L{i}", 0.75, 0.55, Side.NO, close_time=T0 + timedelta(days=i), cluster_id=f"c{i}")
        for i in range(15)
    ]
    result = simulate(candidates, default_params(risk_of_ruin_halt_threshold=0.5, max_drawdown_hard_stop=0.9))
    assert result.halted_at is not None
    assert len(result.trades) < len(candidates)


def test_no_double_counting_when_market_never_settles():
    market = Market(
        ticker="UNSETTLED", event_ticker="EVT", title="t", subtitle="s", status=MarketStatus.OPEN,
        open_time=T0, close_time=T0, expiration_time=T0, settlement_criteria="x", fetched_at=T0,
        result=None,
    )
    ob = make_orderbook(ticker="UNSETTLED", yes_asks=[(55, 1000)])
    estimate = ProbabilityEstimate(market_ticker="UNSETTLED", probability=0.75, variance=0.02,
                                    model_name="t", model_version="1")
    edge = compute_edge(estimate, ob, Side.YES, "EVT", ZERO_FEE_SCHEDULE)
    candidate = TradeCandidate(market=market, decision_time=T0, estimate=estimate, orderbook=ob,
                                edge=edge, cluster_id="c1", category="sports", series_ticker="EVT")
    result = simulate([candidate], default_params())
    assert len(result.trades) == 0
    assert result.final_bankroll == 500.0

from datetime import datetime, timedelta

from kalshi_quant.backtest.sensitivity import run_sensitivity_grid
from kalshi_quant.data.models import Side
from tests.backtest.test_engine_simulate import default_params, make_candidate

T0 = datetime(2024, 1, 1)


def test_sensitivity_grid_covers_all_combinations():
    candidates = [
        make_candidate(f"S{i}", 0.75, 0.55, Side.YES, close_time=T0 + timedelta(days=i), cluster_id=f"c{i}")
        for i in range(5)
    ]
    points = run_sensitivity_grid(
        candidates, default_params(),
        edge_thresholds=[0.02, 0.05], kelly_fraction_caps=[0.25, 0.5],
        fee_multiplier_scales=[1.0, 2.0],
    )
    assert len(points) == 2 * 2 * 2


def test_higher_edge_threshold_never_increases_trade_count():
    candidates = [
        make_candidate(f"T{i}", 0.75, 0.55, Side.YES, close_time=T0 + timedelta(days=i), cluster_id=f"c{i}")
        for i in range(5)
    ]
    points = run_sensitivity_grid(
        candidates, default_params(), edge_thresholds=[0.01, 0.5],
        kelly_fraction_caps=[0.5], fee_multiplier_scales=[1.0],
    )
    low_threshold_trades = next(p.n_trades for p in points if p.min_edge_after_fees == 0.01)
    high_threshold_trades = next(p.n_trades for p in points if p.min_edge_after_fees == 0.5)
    assert high_threshold_trades <= low_threshold_trades


def test_higher_fee_scale_reduces_or_maintains_return():
    candidates = [
        make_candidate(f"F{i}", 0.75, 0.55, Side.YES, close_time=T0 + timedelta(days=i), cluster_id=f"c{i}")
        for i in range(5)
    ]
    points = run_sensitivity_grid(
        candidates, default_params(), edge_thresholds=[0.02],
        kelly_fraction_caps=[0.5], fee_multiplier_scales=[1.0, 10.0],
    )
    low_fee_return = next(p.total_return_pct for p in points if p.fee_multiplier_scale == 1.0)
    high_fee_return = next(p.total_return_pct for p in points if p.fee_multiplier_scale == 10.0)
    assert high_fee_return <= low_fee_return

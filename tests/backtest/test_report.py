from datetime import datetime, timedelta

from kalshi_quant.backtest.engine import SimParams, simulate
from kalshi_quant.backtest.report import build_report, format_report_text, max_drawdown
from kalshi_quant.data.models import Side
from kalshi_quant.risk.portfolio import PortfolioConstraints
from tests.backtest.test_engine_simulate import default_params, make_candidate

T0 = datetime(2024, 1, 1)


def test_max_drawdown_flat_curve_is_zero():
    curve = [(T0, 100.0), (T0 + timedelta(days=1), 100.0)]
    assert max_drawdown(curve) == 0.0


def test_max_drawdown_detects_peak_to_trough():
    curve = [(T0, 100.0), (T0 + timedelta(days=1), 150.0), (T0 + timedelta(days=2), 90.0)]
    assert max_drawdown(curve) == (150 - 90) / 150


def test_build_report_on_winning_trades():
    candidates = [
        make_candidate(f"W{i}", 0.75, 0.55, Side.YES, close_time=T0 + timedelta(days=i), cluster_id=f"c{i}")
        for i in range(10)
    ]
    result = simulate(candidates, default_params())
    report = build_report(result, starting_capital=500.0)
    assert report.final_bankroll > report.starting_capital
    assert report.win_rate == 1.0
    assert report.n_trades == 10
    assert report.calibration is not None
    text = format_report_text(report)
    assert "Backtest Report" in text


def test_build_report_handles_zero_trades():
    result = simulate([], default_params())
    report = build_report(result, starting_capital=500.0)
    assert report.n_trades == 0
    assert report.final_bankroll == 500.0
    assert report.calibration is None

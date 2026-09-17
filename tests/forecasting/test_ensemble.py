import pytest

from kalshi_quant.forecasting.base import ProbabilityEstimate
from kalshi_quant.forecasting.ensemble import ensemble_estimates


def make_estimate(prob, var, ticker="A"):
    return ProbabilityEstimate(market_ticker=ticker, probability=prob, variance=var,
                                model_name="m", model_version="1")


def test_single_estimate_passthrough():
    e = make_estimate(0.7, 0.05)
    result = ensemble_estimates([e])
    assert result.probability == pytest.approx(0.7)
    assert result.variance == pytest.approx(0.05)


def test_ensemble_weights_toward_more_confident_estimate():
    confident = make_estimate(0.9, 0.001)
    unsure = make_estimate(0.5, 0.2)
    result = ensemble_estimates([confident, unsure])
    assert result.probability > 0.7  # pulled strongly toward the confident estimate


def test_ensemble_rejects_mismatched_tickers():
    a = make_estimate(0.6, 0.05, ticker="A")
    b = make_estimate(0.6, 0.05, ticker="B")
    with pytest.raises(ValueError):
        ensemble_estimates([a, b])


def test_ensemble_rejects_empty_list():
    with pytest.raises(ValueError):
        ensemble_estimates([])


def test_ensemble_variance_floor_is_min_of_members():
    a = make_estimate(0.6, 0.05)
    b = make_estimate(0.65, 0.1)
    result = ensemble_estimates([a, b])
    assert result.variance == pytest.approx(0.05)

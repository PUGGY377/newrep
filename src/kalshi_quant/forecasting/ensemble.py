"""Confidence-weighted ensembling of multiple Forecasters.

Combines several ProbabilityEstimates for the same market into one, by
weighting each forecaster's log-odds contribution by its inverse
variance (precision weighting) -- an estimate with high confidence
(low variance) pulls the ensemble more than a high-variance one, and
the ensemble's own reported variance can never be smaller than the best
individual model's, which keeps downstream Kelly shrinkage honest.
"""

from __future__ import annotations

import numpy as np

from kalshi_quant.forecasting.base import ProbabilityEstimate

_EPS = 1e-6


def _logit(p: float) -> float:
    p = min(max(p, _EPS), 1 - _EPS)
    return float(np.log(p / (1 - p)))


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def ensemble_estimates(
    estimates: list[ProbabilityEstimate], model_name: str = "ensemble", model_version: str = "1"
) -> ProbabilityEstimate:
    if not estimates:
        raise ValueError("Cannot ensemble an empty list of estimates")
    tickers = {e.market_ticker for e in estimates}
    if len(tickers) != 1:
        raise ValueError(f"All estimates must be for the same market, got {tickers}")

    if len(estimates) == 1:
        e = estimates[0]
        return ProbabilityEstimate(
            market_ticker=e.market_ticker, probability=e.probability, variance=e.variance,
            model_name=model_name, model_version=model_version,
        )

    weights = np.array([1.0 / max(e.variance, _EPS) for e in estimates])
    weights = weights / weights.sum()
    logits = np.array([_logit(e.probability) for e in estimates])
    combined_logit = float(np.dot(weights, logits))
    combined_probability = _sigmoid(combined_logit)

    # Ensemble variance floor: never below the most confident member's
    # variance. Precision-weighted combination reduces variance when
    # members are independent, but forecasters built on overlapping data
    # sources are not independent -- assuming independence here would
    # systematically overstate confidence, so we deliberately take the
    # conservative floor rather than the (lower) naive inverse-sum.
    combined_variance = float(min(e.variance for e in estimates))

    return ProbabilityEstimate(
        market_ticker=estimates[0].market_ticker, probability=combined_probability,
        variance=combined_variance, model_name=model_name, model_version=model_version,
    )

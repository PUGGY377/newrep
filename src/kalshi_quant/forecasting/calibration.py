"""Calibration: turn a raw model score into a genuinely calibrated
probability, and track how well-calibrated the result actually is.

Two calibration methods are supported:
  - Platt scaling: fits a logistic regression of outcome ~ raw_score.
    Good default when you have limited data or expect a roughly sigmoid
    miscalibration shape.
  - Isotonic regression: fits a non-parametric monotonic map. More
    flexible, needs more data to avoid overfitting the calibration curve
    itself.

Both are fit ONLY on a held-out calibration set that the underlying model
did not train on -- fitting calibration on the training set silently
reintroduces the overfitting calibration is supposed to remove.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


@dataclass
class CalibrationReport:
    brier_score: float
    log_loss_value: float
    n_samples: int
    bin_edges: np.ndarray
    bin_mean_predicted: np.ndarray
    bin_mean_observed: np.ndarray
    bin_counts: np.ndarray


class Calibrator:
    """Wraps a fitted Platt or isotonic mapping. `raw_scores` are whatever
    a Forecaster's underlying model produces before calibration (e.g. an
    Elo win probability, a GBM's predict_proba output) -- not required to
    already be a probability."""

    def __init__(self, method: Literal["platt", "isotonic"] = "isotonic"):
        self.method = method
        self._model: LogisticRegression | IsotonicRegression | None = None
        self._is_fit = False

    def fit(self, raw_scores: np.ndarray, outcomes: np.ndarray) -> None:
        if len(raw_scores) != len(outcomes):
            raise ValueError("raw_scores and outcomes must be the same length")
        if len(raw_scores) < 20:
            raise ValueError(
                f"Refusing to fit calibration on {len(raw_scores)} samples; "
                "need at least 20 held-out examples to get a meaningful curve."
            )
        if self.method == "platt":
            model = LogisticRegression()
            model.fit(raw_scores.reshape(-1, 1), outcomes)
            self._model = model
        else:
            model = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4)
            model.fit(raw_scores, outcomes)
            self._model = model
        self._is_fit = True

    def transform(self, raw_scores: np.ndarray) -> np.ndarray:
        if not self._is_fit or self._model is None:
            raise RuntimeError("Calibrator.fit must be called before transform")
        if self.method == "platt":
            assert isinstance(self._model, LogisticRegression)
            return self._model.predict_proba(raw_scores.reshape(-1, 1))[:, 1]
        assert isinstance(self._model, IsotonicRegression)
        return self._model.predict(raw_scores)

    def transform_one(self, raw_score: float) -> float:
        return float(self.transform(np.array([raw_score]))[0])


def evaluate_calibration(
    predicted_probabilities: np.ndarray, outcomes: np.ndarray, n_bins: int = 10
) -> CalibrationReport:
    """Brier score, log loss, and a reliability-diagram binning, computed
    against realized outcomes. This is the report to render as the
    calibration curve in backtest output and the monitoring dashboard."""
    if len(predicted_probabilities) != len(outcomes):
        raise ValueError("predicted_probabilities and outcomes must be the same length")

    eps = 1e-7
    clipped = np.clip(predicted_probabilities, eps, 1 - eps)
    brier = brier_score_loss(outcomes, clipped)
    ll = log_loss(outcomes, clipped, labels=[0, 1])

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(clipped, bin_edges[1:-1])
    bin_mean_predicted = np.full(n_bins, np.nan)
    bin_mean_observed = np.full(n_bins, np.nan)
    bin_counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = bin_idx == b
        bin_counts[b] = mask.sum()
        if mask.any():
            bin_mean_predicted[b] = clipped[mask].mean()
            bin_mean_observed[b] = outcomes[mask].mean()

    return CalibrationReport(
        brier_score=float(brier),
        log_loss_value=float(ll),
        n_samples=len(outcomes),
        bin_edges=bin_edges,
        bin_mean_predicted=bin_mean_predicted,
        bin_mean_observed=bin_mean_observed,
        bin_counts=bin_counts,
    )

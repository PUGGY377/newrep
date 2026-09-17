"""Forecaster interface.

A Forecaster maps (market, point-in-time feature snapshot) -> a
calibrated probability estimate with an uncertainty measure. The
uncertainty measure is not decorative: risk/kelly.py uses it directly to
shrink position sizing, so a Forecaster that always reports near-zero
variance will get sized as if it were perfectly calibrated, which is
dangerous. Report variance honestly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from kalshi_quant.data.models import FeatureSnapshot, Market


@dataclass(frozen=True)
class ProbabilityEstimate:
    """Output of a Forecaster for one market at one point in time."""

    market_ticker: str
    probability: float  # P(YES), calibrated, in (0, 1)
    variance: float  # estimation uncertainty of `probability` itself, >= 0
    model_name: str
    model_version: str

    def __post_init__(self) -> None:
        if not 0.0 < self.probability < 1.0:
            raise ValueError(f"probability must be in (0, 1), got {self.probability}")
        if self.variance < 0.0:
            raise ValueError(f"variance must be >= 0, got {self.variance}")

    @property
    def confidence(self) -> float:
        """Normalized confidence in [0, 1], derived from variance.

        Uses variance relative to the maximum possible variance of a
        Bernoulli-probability estimate (0.25, at p=0.5) as the
        normalization -- this is a reasonable default, not a universal
        law, and can be overridden by ensembling/calibration code that
        has a better-informed scale for a specific model.
        """
        max_variance = 0.25
        return max(0.0, 1.0 - min(self.variance, max_variance) / max_variance)


class Forecaster(ABC):
    """Concrete implementations: forecasting/sports_elo.py, etc."""

    name: str
    version: str

    @abstractmethod
    def predict(self, market: Market, snapshot: FeatureSnapshot) -> ProbabilityEstimate:
        """Must not use any information dated after snapshot.as_of. The
        snapshot is the ONLY source of external features -- implementations
        must not reach out to a live data source themselves inside this
        method, or point-in-time correctness breaks silently in backtests."""
        raise NotImplementedError

    @abstractmethod
    def fit(self, training_examples: list[tuple[Market, FeatureSnapshot, bool]]) -> None:
        """Train/update the model. `training_examples` is a list of
        (market, snapshot_as_of_training_cutoff, realized_outcome_yes).
        Implementations of walk-forward retraining must ensure every
        snapshot's as_of predates the market's settlement -- the backtest
        engine is responsible for not passing overlapping windows, but a
        defensive check here is cheap insurance."""
        raise NotImplementedError

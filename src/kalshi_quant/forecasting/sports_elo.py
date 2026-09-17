"""Elo-rating sports Forecaster -- the first concrete, end-to-end Forecaster.

Design, so a second Forecaster is easy to add later:
  - `EloRatingEngine` is the actual model: pure function of a chronological
    game log -> ratings at any point in time. It has no knowledge of
    Kalshi, markets, or contracts.
  - `SportsEloForecaster` is the adapter implementing the `Forecaster` ABC:
    it turns a `FeatureSnapshot` (team names) into an Elo win probability,
    then runs it through a `Calibrator` before returning a
    `ProbabilityEstimate`. Any future Forecaster (a GBM classifier, an
    ensemble member) follows the same shape: a category-specific model
    class plus a thin adapter satisfying the ABC.

Point-in-time correctness for training: `fit()` requires each training
example's snapshot.as_of to be strictly before the market's settlement,
and internally replays games in chronological order so a team's rating at
game N only reflects games 1..N-1 -- exactly the walk-forward requirement
the backtest engine depends on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from kalshi_quant.data.models import FeatureSnapshot, Market
from kalshi_quant.forecasting.base import Forecaster, ProbabilityEstimate
from kalshi_quant.forecasting.calibration import Calibrator

DEFAULT_RATING = 1500.0
DEFAULT_K_FACTOR = 20.0
DEFAULT_HOME_ADVANTAGE = 65.0  # Elo points added to the home team's rating pre-game


@dataclass(frozen=True)
class GameResult:
    game_date: datetime
    home_team: str
    away_team: str
    home_won: bool


@dataclass
class EloRatingEngine:
    k_factor: float = DEFAULT_K_FACTOR
    home_advantage: float = DEFAULT_HOME_ADVANTAGE
    _games: list[GameResult] = field(default_factory=list)

    def add_game(self, game: GameResult) -> None:
        self._games.append(game)

    @staticmethod
    def expected_home_win_probability(home_rating: float, away_rating: float, home_advantage: float) -> float:
        diff = (home_rating + home_advantage) - away_rating
        return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

    def ratings_as_of(self, as_of: datetime) -> dict[str, float]:
        """Replay every game strictly before `as_of`, in chronological
        order, to reconstruct each team's rating at that instant. O(n) per
        call -- fine at this system's data scale; cache externally if it
        becomes a bottleneck."""
        ratings: dict[str, float] = {}
        for game in sorted(self._games, key=lambda g: g.game_date):
            if game.game_date >= as_of:
                break
            home_r = ratings.get(game.home_team, DEFAULT_RATING)
            away_r = ratings.get(game.away_team, DEFAULT_RATING)
            expected_home = self.expected_home_win_probability(home_r, away_r, self.home_advantage)
            actual_home = 1.0 if game.home_won else 0.0
            ratings[game.home_team] = home_r + self.k_factor * (actual_home - expected_home)
            ratings[game.away_team] = away_r + self.k_factor * ((1 - actual_home) - (1 - expected_home))
        return ratings

    def predict_home_win_probability(self, home_team: str, away_team: str, as_of: datetime) -> float:
        ratings = self.ratings_as_of(as_of)
        home_r = ratings.get(home_team, DEFAULT_RATING)
        away_r = ratings.get(away_team, DEFAULT_RATING)
        return self.expected_home_win_probability(home_r, away_r, self.home_advantage)


class SportsEloForecaster(Forecaster):
    name = "sports_elo"
    version = "0.1.0"

    def __init__(self, k_factor: float = DEFAULT_K_FACTOR, home_advantage: float = DEFAULT_HOME_ADVANTAGE,
                 calibration_method: str = "isotonic"):
        self.engine = EloRatingEngine(k_factor=k_factor, home_advantage=home_advantage)
        self.calibrator = Calibrator(method=calibration_method)  # type: ignore[arg-type]
        self._is_calibrated = False

    def fit(self, training_examples: list[tuple[Market, FeatureSnapshot, bool]]) -> None:
        sorted_examples = sorted(training_examples, key=lambda ex: ex[1].as_of)

        for market, snapshot, outcome_yes in sorted_examples:
            if market.settled_at is not None and snapshot.as_of >= market.settled_at:
                raise ValueError(
                    f"Training example for {market.ticker} has snapshot.as_of "
                    f"({snapshot.as_of}) >= settlement time ({market.settled_at}); "
                    "this would leak the outcome into training."
                )
            home_team = snapshot.values.get("home_team")
            away_team = snapshot.values.get("away_team")
            yes_team = snapshot.values.get("yes_team")
            if not (home_team and away_team and yes_team):
                continue
            home_won = outcome_yes if yes_team == home_team else not outcome_yes
            self.engine.add_game(
                GameResult(
                    game_date=snapshot.as_of, home_team=str(home_team), away_team=str(away_team),
                    home_won=bool(home_won),
                )
            )

        # Fit calibration on raw Elo probabilities vs. realized outcomes,
        # using each example's rating state AS OF that game (i.e. excluding
        # the game itself) -- computed by replaying up to but not including
        # each game in turn, consistent with the walk-forward contract.
        raw_scores = []
        outcomes = []
        replay_engine = EloRatingEngine(k_factor=self.engine.k_factor, home_advantage=self.engine.home_advantage)
        for market, snapshot, outcome_yes in sorted_examples:
            home_team = snapshot.values.get("home_team")
            away_team = snapshot.values.get("away_team")
            yes_team = snapshot.values.get("yes_team")
            if not (home_team and away_team and yes_team):
                continue
            p_home = replay_engine.predict_home_win_probability(str(home_team), str(away_team), snapshot.as_of)
            p_yes = p_home if yes_team == home_team else 1 - p_home
            raw_scores.append(p_yes)
            outcomes.append(1.0 if outcome_yes else 0.0)
            home_won = outcome_yes if yes_team == home_team else not outcome_yes
            replay_engine.add_game(
                GameResult(game_date=snapshot.as_of, home_team=str(home_team), away_team=str(away_team),
                           home_won=bool(home_won))
            )

        if len(raw_scores) >= 20:
            self.calibrator.fit(np.array(raw_scores), np.array(outcomes))
            self._is_calibrated = True

    def predict(self, market: Market, snapshot: FeatureSnapshot) -> ProbabilityEstimate:
        home_team = snapshot.values.get("home_team")
        away_team = snapshot.values.get("away_team")
        yes_team = snapshot.values.get("yes_team")
        if not (home_team and away_team and yes_team):
            raise ValueError(f"Snapshot for {market.ticker} missing home_team/away_team/yes_team")

        p_home = self.engine.predict_home_win_probability(str(home_team), str(away_team), snapshot.as_of)
        p_yes_raw = p_home if yes_team == home_team else 1 - p_home

        if self._is_calibrated:
            p_yes = self.calibrator.transform_one(p_yes_raw)
        else:
            p_yes = p_yes_raw

        p_yes = min(max(p_yes, 1e-4), 1 - 1e-4)

        # Uncertainty proxy: distance from 0.5 correlates with how much
        # rating data actually separates the two teams; a pair of teams
        # with identical ratings and a coin-flip Elo probability is exactly
        # the case where the model has the least information, so variance
        # is highest there. This is a documented heuristic, not a fitted
        # uncertainty model -- it should be replaced with an empirical
        # variance estimate (e.g. from calibration residuals) once there's
        # enough live history to fit one reliably.
        variance = 0.25 * (1.0 - abs(p_yes_raw - 0.5) * 2) ** 0.5 * 0.25 + 1e-4

        return ProbabilityEstimate(
            market_ticker=market.ticker, probability=p_yes, variance=variance,
            model_name=self.name, model_version=self.version,
        )

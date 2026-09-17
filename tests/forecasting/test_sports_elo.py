from datetime import datetime, timedelta

import pytest

from kalshi_quant.data.models import FeatureSnapshot, Market, MarketStatus, Side
from kalshi_quant.forecasting.sports_elo import (
    DEFAULT_RATING,
    EloRatingEngine,
    GameResult,
    SportsEloForecaster,
)

T0 = datetime(2024, 1, 1)


def make_market(ticker="KXNFLGAME-A", settled_at=None) -> Market:
    return Market(
        ticker=ticker, event_ticker="KXNFLGAME-A-EVT", title="t", subtitle="s",
        status=MarketStatus.SETTLED if settled_at else MarketStatus.OPEN,
        open_time=T0, close_time=T0, expiration_time=T0, settlement_criteria="x",
        fetched_at=T0, settled_at=settled_at,
    )


def test_new_teams_start_at_default_rating():
    engine = EloRatingEngine()
    ratings = engine.ratings_as_of(T0)
    assert ratings == {}
    p = engine.predict_home_win_probability("A", "B", T0)
    # home advantage should make home team slightly favored even with equal base ratings
    assert p > 0.5


def test_ratings_as_of_excludes_future_games():
    engine = EloRatingEngine()
    engine.add_game(GameResult(game_date=T0, home_team="A", away_team="B", home_won=True))
    engine.add_game(GameResult(game_date=T0 + timedelta(days=10), home_team="A", away_team="B", home_won=True))

    ratings_before_second_game = engine.ratings_as_of(T0 + timedelta(days=5))
    ratings_after_both = engine.ratings_as_of(T0 + timedelta(days=20))

    assert ratings_before_second_game["A"] != ratings_after_both["A"]
    # A won game 1 -> rating should have gone up from default
    assert ratings_before_second_game["A"] > DEFAULT_RATING


def test_winner_rating_increases_loser_decreases():
    engine = EloRatingEngine()
    engine.add_game(GameResult(game_date=T0, home_team="A", away_team="B", home_won=True))
    ratings = engine.ratings_as_of(T0 + timedelta(days=1))
    assert ratings["A"] > DEFAULT_RATING
    assert ratings["B"] < DEFAULT_RATING


def test_forecaster_fit_rejects_leaked_training_example():
    forecaster = SportsEloForecaster()
    settled_market = make_market(settled_at=T0)
    snapshot = FeatureSnapshot(
        market_ticker=settled_market.ticker, as_of=T0 + timedelta(hours=1),  # AFTER settlement
        values={"home_team": "A", "away_team": "B", "yes_team": "A"},
    )
    with pytest.raises(ValueError):
        forecaster.fit([(settled_market, snapshot, True)])


def test_forecaster_predict_requires_team_features():
    forecaster = SportsEloForecaster()
    market = make_market()
    bad_snapshot = FeatureSnapshot(market_ticker=market.ticker, as_of=T0, values={})
    with pytest.raises(ValueError):
        forecaster.predict(market, bad_snapshot)


def test_forecaster_end_to_end_fit_and_predict():
    forecaster = SportsEloForecaster()
    examples = []
    for i in range(30):
        market = make_market(ticker=f"KXNFLGAME-{i}", settled_at=T0 + timedelta(days=i, hours=3))
        snapshot = FeatureSnapshot(
            market_ticker=market.ticker, as_of=T0 + timedelta(days=i),
            values={"home_team": "STRONG", "away_team": "WEAK", "yes_team": "STRONG"},
        )
        examples.append((market, snapshot, True))  # STRONG always wins historically

    forecaster.fit(examples)

    next_market = make_market(ticker="KXNFLGAME-NEXT")
    next_snapshot = FeatureSnapshot(
        market_ticker=next_market.ticker, as_of=T0 + timedelta(days=31),
        values={"home_team": "STRONG", "away_team": "WEAK", "yes_team": "STRONG"},
    )
    estimate = forecaster.predict(next_market, next_snapshot)
    assert estimate.probability > 0.5
    assert 0.0 <= estimate.variance <= 0.25

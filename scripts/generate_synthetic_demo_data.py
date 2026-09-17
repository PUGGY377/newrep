#!/usr/bin/env python
"""Generate a SYNTHETIC sports market dataset for exercising the backtest
engine end-to-end before real collected Kalshi history exists (see the
limitation documented in ingest_historical.py -- historical order-book
depth isn't retroactively available from Kalshi's API).

This is explicitly fake data: two teams with a fixed true skill gap play
many games; the "market" price is a noisy, slightly-biased proxy for the
true win probability (simulating the kind of soft mispricing an edge
strategy would try to exploit), and outcomes are drawn from the true
probability. It exists ONLY to prove the pipeline (data -> forecaster ->
edge -> sizing -> backtest -> report) runs correctly and produces a
sane-looking report -- it is NOT evidence the strategy is profitable on
real markets. Do not use its output to size real capital.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kalshi_quant.data.models import Market, MarketStatus, OrderbookLevel, OrderbookSnapshot, Side
from kalshi_quant.data.store import DataStore

RNG = np.random.default_rng(7)


def true_win_probability(home_rating: float, away_rating: float) -> float:
    diff = (home_rating + 65.0) - away_rating
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


def generate(db_path: str, n_games: int = 400) -> None:
    store = DataStore(db_path)
    teams = [f"TEAM{i}" for i in range(8)]
    true_ratings = {t: RNG.normal(1500, 120) for t in teams}

    start = datetime(2023, 9, 1, tzinfo=timezone.utc)
    for i in range(n_games):
        home, away = RNG.choice(teams, size=2, replace=False)
        game_time = start + timedelta(days=i * 2)
        true_p = true_win_probability(true_ratings[home], true_ratings[away])
        home_won = RNG.uniform() < true_p
        result = Side.YES  # market is "does home team win?" -> YES means home won

        ticker = f"KXNFLGAME-{i:04d}"
        market = Market(
            ticker=ticker, event_ticker=f"{ticker}-EVT",
            title=f"{away} at {home}", subtitle=home,
            status=MarketStatus.SETTLED,
            open_time=game_time - timedelta(days=3), close_time=game_time,
            expiration_time=game_time, settlement_criteria="Home team wins",
            result=Side.YES if home_won else Side.NO, settled_at=game_time,
            fetched_at=game_time - timedelta(days=3),
        )
        store.write_market(market)

        # Market price: a NOISY, slightly-biased proxy for true_p, simulating
        # a market that's directionally right but not perfectly efficient --
        # this is what gives a well-calibrated model room for edge.
        market_bias = -0.04  # market slightly underprices the home team
        noisy_price = np.clip(true_p + market_bias + RNG.normal(0, 0.05), 0.05, 0.95)
        yes_price_cents = int(round(noisy_price * 100))
        spread = 2
        yes_bid = max(1, yes_price_cents - spread // 2)
        yes_ask = min(99, yes_price_cents + spread // 2)

        decision_time = game_time - timedelta(hours=6)
        store.write_orderbook_snapshot(OrderbookSnapshot(
            market_ticker=ticker, as_of=decision_time,
            yes_bids=(OrderbookLevel(yes_bid, 500),), yes_asks=(OrderbookLevel(yes_ask, 500),),
            no_bids=(OrderbookLevel(100 - yes_ask, 500),), no_asks=(OrderbookLevel(100 - yes_bid, 500),),
        ))

    store.close()
    print(f"Generated {n_games} synthetic settled markets into {db_path}")


if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else "synthetic_demo.duckdb"
    generate(db_path)

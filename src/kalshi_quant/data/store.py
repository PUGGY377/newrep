"""Point-in-time local storage (DuckDB).

The single rule this module exists to enforce: every read that will feed
a backtest decision at simulated time T must only see rows whose
`as_of`/`fetched_at` timestamp is <= T. Every "as_of" query method takes
an explicit `as_of` cutoff and pushes it into the SQL WHERE clause --
there is no method that returns "the latest snapshot" without a cutoff,
specifically so a backtest loop can never accidentally call one and leak
the future into the past. Live/paper trading always passes
`as_of=utc_now()`, which is just the trivial case of the same rule.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

from kalshi_quant.data.models import (
    Event,
    FeatureSnapshot,
    Market,
    MarketStatus,
    OrderbookLevel,
    OrderbookSnapshot,
    Series,
    Side,
    Trade,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS series (
    ticker TEXT NOT NULL,
    title TEXT,
    category TEXT,
    fetched_at TIMESTAMP NOT NULL,
    PRIMARY KEY (ticker, fetched_at)
);

CREATE TABLE IF NOT EXISTS events (
    event_ticker TEXT NOT NULL,
    series_ticker TEXT,
    title TEXT,
    strike_date TIMESTAMP,
    fetched_at TIMESTAMP NOT NULL,
    PRIMARY KEY (event_ticker, fetched_at)
);

CREATE TABLE IF NOT EXISTS markets (
    ticker TEXT NOT NULL,
    event_ticker TEXT,
    title TEXT,
    subtitle TEXT,
    status TEXT,
    open_time TIMESTAMP,
    close_time TIMESTAMP,
    expiration_time TIMESTAMP,
    settlement_criteria TEXT,
    result TEXT,
    settled_at TIMESTAMP,
    fetched_at TIMESTAMP NOT NULL,
    PRIMARY KEY (ticker, fetched_at)
);

CREATE TABLE IF NOT EXISTS orderbook_snapshots (
    market_ticker TEXT NOT NULL,
    as_of TIMESTAMP NOT NULL,
    yes_bids JSON,
    yes_asks JSON,
    no_bids JSON,
    no_asks JSON,
    PRIMARY KEY (market_ticker, as_of)
);

CREATE TABLE IF NOT EXISTS trades (
    market_ticker TEXT NOT NULL,
    executed_at TIMESTAMP NOT NULL,
    yes_price_cents INTEGER,
    count INTEGER,
    taker_side TEXT,
    PRIMARY KEY (market_ticker, executed_at, yes_price_cents, count)
);

CREATE TABLE IF NOT EXISTS feature_snapshots (
    market_ticker TEXT NOT NULL,
    as_of TIMESTAMP NOT NULL,
    values_json JSON,
    sources_json JSON,
    PRIMARY KEY (market_ticker, as_of)
);

CREATE TABLE IF NOT EXISTS forecasts (
    market_ticker TEXT NOT NULL,
    as_of TIMESTAMP NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    probability DOUBLE,
    variance DOUBLE,
    PRIMARY KEY (market_ticker, as_of, model_name, model_version)
);
"""


class DataStore:
    """DuckDB-backed point-in-time store. Swap to Postgres later by
    replacing this class -- callers only depend on the methods below."""

    def __init__(self, db_path: str | Path = "kalshi_quant.duckdb"):
        self._conn = duckdb.connect(str(db_path))
        self._conn.execute(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "DataStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- writes: append-only, never UPDATE -------------------------------
    # Every write is a new row keyed by (entity, fetched_at/as_of). We never
    # UPDATE a market/event row in place: Kalshi metadata (e.g. status,
    # result) changes over time, and overwriting would destroy the point-in-
    # time history a backtest needs. Query methods always select the latest
    # row <= as_of instead.

    def write_series(self, series: Series) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO series VALUES (?, ?, ?, ?)",
            [series.ticker, series.title, series.category, series.fetched_at],
        )

    def write_event(self, event: Event) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO events VALUES (?, ?, ?, ?, ?)",
            [event.event_ticker, event.series_ticker, event.title, event.strike_date,
             event.fetched_at],
        )

    def write_market(self, market: Market) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO markets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                market.ticker, market.event_ticker, market.title, market.subtitle,
                market.status.value, market.open_time, market.close_time,
                market.expiration_time, market.settlement_criteria,
                market.result.value if market.result else None, market.settled_at,
                market.fetched_at,
            ],
        )

    def write_orderbook_snapshot(self, snapshot: OrderbookSnapshot) -> None:
        import json

        def levels_json(levels: tuple[OrderbookLevel, ...]) -> str:
            return json.dumps([{"price_cents": l.price_cents, "contracts": l.contracts} for l in levels])

        self._conn.execute(
            "INSERT OR REPLACE INTO orderbook_snapshots VALUES (?, ?, ?, ?, ?, ?)",
            [
                snapshot.market_ticker, snapshot.as_of,
                levels_json(snapshot.yes_bids), levels_json(snapshot.yes_asks),
                levels_json(snapshot.no_bids), levels_json(snapshot.no_asks),
            ],
        )

    def write_trade(self, trade: Trade) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO trades VALUES (?, ?, ?, ?, ?)",
            [trade.market_ticker, trade.executed_at, trade.yes_price_cents, trade.count,
             trade.taker_side.value],
        )

    def write_feature_snapshot(self, snapshot: FeatureSnapshot) -> None:
        import json

        self._conn.execute(
            "INSERT OR REPLACE INTO feature_snapshots VALUES (?, ?, ?, ?)",
            [
                snapshot.market_ticker, snapshot.as_of,
                json.dumps(snapshot.values), json.dumps(snapshot.sources),
            ],
        )

    def write_forecast(self, market_ticker: str, as_of: datetime, model_name: str,
                        model_version: str, probability: float, variance: float) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO forecasts VALUES (?, ?, ?, ?, ?, ?)",
            [market_ticker, as_of, model_name, model_version, probability, variance],
        )

    # -- point-in-time reads ----------------------------------------------

    def get_market_as_of(self, ticker: str, as_of: datetime) -> Market | None:
        row = self._conn.execute(
            "SELECT * FROM markets WHERE ticker = ? AND fetched_at <= ? "
            "ORDER BY fetched_at DESC LIMIT 1",
            [ticker, as_of],
        ).fetchone()
        if row is None:
            return None
        cols = [d[0] for d in self._conn.description]
        d = dict(zip(cols, row))
        return Market(
            ticker=d["ticker"], event_ticker=d["event_ticker"], title=d["title"],
            subtitle=d["subtitle"], status=MarketStatus(d["status"]), open_time=d["open_time"],
            close_time=d["close_time"], expiration_time=d["expiration_time"],
            settlement_criteria=d["settlement_criteria"],
            result=Side(d["result"]) if d["result"] else None, settled_at=d["settled_at"],
            fetched_at=d["fetched_at"],
        )

    def get_orderbook_as_of(self, market_ticker: str, as_of: datetime) -> OrderbookSnapshot | None:
        import json

        row = self._conn.execute(
            "SELECT * FROM orderbook_snapshots WHERE market_ticker = ? AND as_of <= ? "
            "ORDER BY as_of DESC LIMIT 1",
            [market_ticker, as_of],
        ).fetchone()
        if row is None:
            return None
        cols = [d[0] for d in self._conn.description]
        d = dict(zip(cols, row))

        def parse_levels(raw: str) -> tuple[OrderbookLevel, ...]:
            return tuple(OrderbookLevel(**lvl) for lvl in json.loads(raw))

        return OrderbookSnapshot(
            market_ticker=d["market_ticker"], as_of=d["as_of"],
            yes_bids=parse_levels(d["yes_bids"]), yes_asks=parse_levels(d["yes_asks"]),
            no_bids=parse_levels(d["no_bids"]), no_asks=parse_levels(d["no_asks"]),
        )

    def get_trades_between(self, market_ticker: str, start: datetime, end: datetime) -> list[Trade]:
        rows = self._conn.execute(
            "SELECT * FROM trades WHERE market_ticker = ? AND executed_at >= ? AND executed_at <= ? "
            "ORDER BY executed_at ASC",
            [market_ticker, start, end],
        ).fetchall()
        cols = [d[0] for d in self._conn.description]
        return [
            Trade(
                market_ticker=d["market_ticker"], executed_at=d["executed_at"],
                yes_price_cents=d["yes_price_cents"], count=d["count"],
                taker_side=Side(d["taker_side"]),
            )
            for d in (dict(zip(cols, row)) for row in rows)
        ]

    def get_feature_snapshot_as_of(self, market_ticker: str, as_of: datetime) -> FeatureSnapshot | None:
        import json

        row = self._conn.execute(
            "SELECT * FROM feature_snapshots WHERE market_ticker = ? AND as_of <= ? "
            "ORDER BY as_of DESC LIMIT 1",
            [market_ticker, as_of],
        ).fetchone()
        if row is None:
            return None
        cols = [d[0] for d in self._conn.description]
        d = dict(zip(cols, row))
        return FeatureSnapshot(
            market_ticker=d["market_ticker"], as_of=d["as_of"],
            values=json.loads(d["values_json"]), sources=json.loads(d["sources_json"]),
        )

    def get_settled_markets(self, event_ticker: str | None = None) -> list[Market]:
        """Latest-known row per ticker where status='settled' -- used by the
        backtester/forecaster training loop to build labeled outcome data."""
        query = (
            "SELECT m.* FROM markets m "
            "INNER JOIN (SELECT ticker, MAX(fetched_at) AS max_fetched FROM markets "
            "GROUP BY ticker) latest "
            "ON m.ticker = latest.ticker AND m.fetched_at = latest.max_fetched "
            "WHERE m.status = 'settled'"
        )
        params: list[Any] = []
        if event_ticker:
            query += " AND m.event_ticker = ?"
            params.append(event_ticker)
        rows = self._conn.execute(query, params).fetchall()
        cols = [d[0] for d in self._conn.description]
        results = []
        for d in (dict(zip(cols, row)) for row in rows):
            results.append(
                Market(
                    ticker=d["ticker"], event_ticker=d["event_ticker"], title=d["title"],
                    subtitle=d["subtitle"], status=MarketStatus(d["status"]),
                    open_time=d["open_time"], close_time=d["close_time"],
                    expiration_time=d["expiration_time"],
                    settlement_criteria=d["settlement_criteria"],
                    result=Side(d["result"]) if d["result"] else None,
                    settled_at=d["settled_at"], fetched_at=d["fetched_at"],
                )
            )
        return results

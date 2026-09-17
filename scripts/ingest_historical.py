#!/usr/bin/env python
"""Pull historical market metadata, orderbook snapshots, and trades from
Kalshi into the local DataStore, for series specified in config.

Usage:
    python scripts/ingest_historical.py --series-ticker KXNFLGAME --days-back 180

Requires KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH to be set (see
.env.example). Uses whatever KALSHI_ENVIRONMENT is set to (demo by
default) -- ingesting historical market data does not require prod
access and demo data is sufficient for building/testing the pipeline,
though only prod has the real trading history you'd eventually backtest
against seriously.

IMPORTANT LIMITATION -- read before trusting a backtest built on this:
Kalshi's REST API does not expose historical order-book depth for past
timestamps; `get_orderbook()` only returns the CURRENT book. For an
already-settled market, this script can only record whatever the book
looks like right now (typically empty/irrelevant for a long-settled
market), not the spread that was live at decision time historically.
Genuine point-in-time order-book history can only be built going
FORWARD, by running this ingestion (or a WebSocket subscriber) on a
schedule from today onward. Trade history (`get_trades`) IS available
historically and is a reasonable proxy for realistic execution prices
in the meantime -- consider extending this script to backfill trades
and approximate a spread around each trade price if you need a backtest
deeper than your own collection window. The synthetic demo dataset in
scripts/generate_synthetic_demo_data.py exists specifically because of
this limitation, to let the backtest engine be exercised end-to-end
before real collected history has accumulated.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import click
import structlog

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kalshi_quant.data.kalshi_client import KalshiClient
from kalshi_quant.data.models import Event, Market, MarketStatus, OrderbookLevel, OrderbookSnapshot, Series, Side
from kalshi_quant.data.store import DataStore

logger = structlog.get_logger(__name__)


def _parse_market(raw: dict, fetched_at: datetime) -> Market:
    result = raw.get("result")
    return Market(
        ticker=raw["ticker"], event_ticker=raw["event_ticker"], title=raw.get("title", ""),
        subtitle=raw.get("yes_sub_title", raw.get("subtitle", "")),
        status=MarketStatus(raw.get("status", "unopened")),
        open_time=datetime.fromisoformat(raw["open_time"]) if raw.get("open_time") else fetched_at,
        close_time=datetime.fromisoformat(raw["close_time"]) if raw.get("close_time") else fetched_at,
        expiration_time=datetime.fromisoformat(raw["expiration_time"]) if raw.get("expiration_time") else fetched_at,
        settlement_criteria=raw.get("rules_primary", ""),
        result=Side(result) if result in ("yes", "no") else None,
        settled_at=datetime.fromisoformat(raw["settlement_time"]) if raw.get("settlement_time") else None,
        fetched_at=fetched_at,
    )


@click.command()
@click.option("--series-ticker", required=True, help="Kalshi series ticker, e.g. KXNFLGAME")
@click.option("--days-back", default=180, help="How many days of history to pull")
@click.option("--db-path", default="kalshi_quant.duckdb")
def main(series_ticker: str, days_back: int, db_path: str) -> None:
    client = KalshiClient()
    store = DataStore(db_path)
    fetched_at = datetime.now(timezone.utc)
    cutoff = fetched_at - timedelta(days=days_back)

    series_raw = client.get_series(series_ticker)
    store.write_series(Series(
        ticker=series_raw["ticker"], title=series_raw.get("title", ""),
        category=series_raw.get("category", ""), fetched_at=fetched_at,
    ))

    n_markets = 0
    for event_raw in client.get_events(series_ticker=series_ticker, status="settled"):
        strike_date = (
            datetime.fromisoformat(event_raw["strike_date"]) if event_raw.get("strike_date") else None
        )
        store.write_event(Event(
            event_ticker=event_raw["event_ticker"], series_ticker=series_ticker,
            title=event_raw.get("title", ""), strike_date=strike_date, fetched_at=fetched_at,
        ))

        for market_raw in client.get_markets(event_ticker=event_raw["event_ticker"]):
            market = _parse_market(market_raw, fetched_at)
            if market.close_time < cutoff:
                continue
            store.write_market(market)
            n_markets += 1

            try:
                book_raw = client.get_orderbook(market.ticker)
            except Exception as e:
                logger.warning("orderbook_fetch_failed", ticker=market.ticker, error=str(e))
                continue
            book = book_raw.get("orderbook", book_raw)
            yes_levels = tuple(OrderbookLevel(price_cents=p, contracts=c) for p, c in book.get("yes", []))
            no_levels = tuple(OrderbookLevel(price_cents=p, contracts=c) for p, c in book.get("no", []))
            store.write_orderbook_snapshot(OrderbookSnapshot(
                market_ticker=market.ticker, as_of=fetched_at, yes_bids=yes_levels,
                yes_asks=tuple(OrderbookLevel(100 - l.price_cents, l.contracts) for l in no_levels),
                no_bids=no_levels,
                no_asks=tuple(OrderbookLevel(100 - l.price_cents, l.contracts) for l in yes_levels),
            ))

    logger.info("ingest_complete", series_ticker=series_ticker, markets_ingested=n_markets)
    store.close()


if __name__ == "__main__":
    main()

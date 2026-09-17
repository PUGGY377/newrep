"""Core data models shared across layers.

Every model that represents something observed over time carries an
`as_of` (or `observed_at`) timestamp distinct from any "effective" or
"settlement" timestamp the underlying event may have. `as_of` records
when the system actually learned the value -- this is the field the
backtester joins on, and it is what makes point-in-time correctness
enforceable rather than aspirational.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class MarketStatus(str, Enum):
    UNOPENED = "unopened"
    OPEN = "open"
    CLOSED = "closed"
    SETTLED = "settled"


class Side(str, Enum):
    YES = "yes"
    NO = "no"


@dataclass(frozen=True)
class Series:
    """A Kalshi series, e.g. 'KXNFLGAME' -- a family of related event contracts."""

    ticker: str
    title: str
    category: str
    fetched_at: datetime


@dataclass(frozen=True)
class Event:
    """A Kalshi event, e.g. a single NFL game -- groups one or more markets."""

    event_ticker: str
    series_ticker: str
    title: str
    strike_date: datetime | None
    fetched_at: datetime


@dataclass(frozen=True)
class Market:
    """A single tradeable binary/categorical contract."""

    ticker: str
    event_ticker: str
    title: str
    subtitle: str
    status: MarketStatus
    open_time: datetime
    close_time: datetime
    expiration_time: datetime
    settlement_criteria: str
    fetched_at: datetime
    result: Side | None = None
    settled_at: datetime | None = None


@dataclass(frozen=True)
class OrderbookLevel:
    price_cents: int  # 1-99
    contracts: int


@dataclass(frozen=True)
class OrderbookSnapshot:
    """Full order book snapshot for one market at one point in time.

    `as_of` is the timestamp to use for all point-in-time joins -- it is
    when this snapshot was captured/received, not when it might be
    processed later.
    """

    market_ticker: str
    as_of: datetime
    yes_bids: tuple[OrderbookLevel, ...]
    yes_asks: tuple[OrderbookLevel, ...]
    no_bids: tuple[OrderbookLevel, ...]
    no_asks: tuple[OrderbookLevel, ...]

    @property
    def best_yes_bid(self) -> int | None:
        return self.yes_bids[0].price_cents if self.yes_bids else None

    @property
    def best_yes_ask(self) -> int | None:
        return self.yes_asks[0].price_cents if self.yes_asks else None

    @property
    def mid_price_cents(self) -> float | None:
        bid, ask = self.best_yes_bid, self.best_yes_ask
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2.0

    @property
    def implied_probability(self) -> float | None:
        """Market-implied probability of YES from the mid price, in [0, 1]."""
        mid = self.mid_price_cents
        return None if mid is None else mid / 100.0


@dataclass(frozen=True)
class Trade:
    market_ticker: str
    executed_at: datetime
    yes_price_cents: int
    count: int
    taker_side: Side


@dataclass(frozen=True)
class FeatureSnapshot:
    """A bundle of point-in-time features for one market, assembled by
    external data connectors, that a Forecaster consumes.

    `as_of` bounds every feature inside `values` -- callers assembling
    this snapshot must not include any datum whose own timestamp is
    after `as_of`. Enforcing that is the connector's responsibility
    (see data/connectors/base.py).
    """

    market_ticker: str
    as_of: datetime
    values: dict[str, float | str | None] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)  # feature name -> source id, for audit

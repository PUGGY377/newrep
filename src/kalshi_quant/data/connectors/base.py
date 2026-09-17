"""Pluggable external data connector interface.

Each market category (sports, macro, weather, politics, ...) attaches its
own connector implementation. The one method every connector must
implement, `fetch_point_in_time`, takes an explicit `as_of` cutoff and
must guarantee that nothing it returns could only have been known after
that instant -- this is the single most important contract in the whole
system, because a violation here silently corrupts every backtest result
downstream without raising any error.

Concretely, implementers must:
  1. Never call a "latest" endpoint of an upstream source without also
     checking that source's own publish/effective timestamp against
     `as_of` and discarding anything published later.
  2. Prefer upstream APIs that expose historical snapshots (e.g. an odds
     API's closing-line history, a polling aggregator's dated releases)
     over "current value" endpoints when building historical
     FeatureSnapshots for backtesting.
  3. Stamp every returned feature with its true source publish time in
     FeatureSnapshot.sources so this can be audited later.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from kalshi_quant.data.models import FeatureSnapshot, Market


class ExternalDataConnector(ABC):
    """One connector instance is scoped to one market category."""

    category: str

    @abstractmethod
    def fetch_point_in_time(self, market: Market, as_of: datetime) -> FeatureSnapshot:
        """Return the features knowable as of `as_of` for `market`.

        Must raise `DataNotAvailableError` (not return a stale/partial
        snapshot silently) if the connector cannot honor the point-in-time
        guarantee for this request -- e.g. its upstream source only offers
        a live/current endpoint and `as_of` is in the past.
        """
        raise NotImplementedError

    @abstractmethod
    def relevant_to(self, market: Market) -> bool:
        """Whether this connector has anything to say about `market` at all
        (e.g. a sports connector matching on event_ticker league prefix)."""
        raise NotImplementedError


class DataNotAvailableError(RuntimeError):
    """Raised instead of returning a snapshot that can't honor the
    point-in-time contract. Callers (forecasters, backtest engine) must
    treat this as 'no signal', not fall back to a live/current value."""

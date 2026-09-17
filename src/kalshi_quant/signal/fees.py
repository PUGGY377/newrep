"""Kalshi fee schedule.

IMPORTANT -- CONFIRM BEFORE TRADING WITH REAL MONEY: Kalshi's fee
schedule is not exposed by a stable, documented "get current fees" REST
endpoint as of this writing, and it has changed more than once
historically (both the multiplier and which markets get a reduced rate).
The formula below matches Kalshi's publicly documented general-market
fee formula at the time this module was written:

    fee_dollars = ceil_to_cent(fee_multiplier * contracts * price * (1 - price))

where `price` is the traded price expressed as a probability in [0, 1]
(i.e. price_cents / 100). This is charged on both resting (maker) and
taking (taker) fills for most markets; Kalshi has introduced reduced or
zero maker fees for specific series in the past, which this module does
NOT model automatically. Before relying on this for sizing decisions
against real capital:
  1. Re-check the current fee schedule in the Kalshi UI/docs for the
     specific series you intend to trade.
  2. Update `FEE_SCHEDULES` below with a new versioned entry rather than
     mutating the existing one, so backtests run against historical data
     keep using the fee schedule that was actually in effect then.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class FeeSchedule:
    version: str
    effective_from: datetime
    effective_to: datetime | None  # None means "still in effect"
    general_multiplier: float
    reduced_multiplier: float | None = None
    reduced_series_prefixes: tuple[str, ...] = ()

    def multiplier_for(self, series_ticker: str) -> float:
        if self.reduced_multiplier is not None and any(
            series_ticker.startswith(p) for p in self.reduced_series_prefixes
        ):
            return self.reduced_multiplier
        return self.general_multiplier

    def fee_dollars(self, series_ticker: str, contracts: int, price_probability: float) -> float:
        if not 0.0 <= price_probability <= 1.0:
            raise ValueError(f"price_probability must be in [0, 1], got {price_probability}")
        if contracts < 0:
            raise ValueError(f"contracts must be >= 0, got {contracts}")
        multiplier = self.multiplier_for(series_ticker)
        raw = multiplier * contracts * price_probability * (1.0 - price_probability)
        return math.ceil(raw * 100) / 100.0  # round up to the nearest cent


# Versioned schedules, oldest first. Backtests select the schedule whose
# [effective_from, effective_to) window contains the trade's timestamp;
# live/paper trading always uses the last entry (effective_to=None).
FEE_SCHEDULES: list[FeeSchedule] = [
    FeeSchedule(
        version="2024-general-0.07",
        effective_from=datetime(2023, 1, 1),
        effective_to=None,
        general_multiplier=0.07,
    ),
]


def fee_schedule_as_of(as_of: datetime) -> FeeSchedule:
    applicable = [
        s for s in FEE_SCHEDULES
        if s.effective_from <= as_of and (s.effective_to is None or as_of < s.effective_to)
    ]
    if not applicable:
        raise ValueError(f"No fee schedule covers {as_of}; add one to FEE_SCHEDULES")
    return applicable[-1]


def current_fee_schedule() -> FeeSchedule:
    return FEE_SCHEDULES[-1]

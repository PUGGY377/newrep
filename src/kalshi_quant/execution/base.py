"""Execution layer interface.

Every concrete engine (paper, live) implements the same interface so the
decision-making code upstream (edge -> size -> order) never has to know
which one it's talking to. The only place that decision is made is
scripts/run_paper_trading.py (or a future run_live_trading.py), which
picks the engine based on config + the double confirmation gate in
config.is_live_trading_authorized -- see execution/live.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from kalshi_quant.data.models import Side


class OrderAction(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    PENDING = "pending"
    RESTING = "resting"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELED = "canceled"
    REJECTED = "rejected"


@dataclass(frozen=True)
class Order:
    client_order_id: str
    market_ticker: str
    side: Side
    action: OrderAction
    count: int
    limit_price_cents: int
    created_at: datetime


@dataclass(frozen=True)
class Fill:
    client_order_id: str
    market_ticker: str
    side: Side
    count: int
    price_cents: int
    filled_at: datetime


@dataclass
class OrderResult:
    order: Order
    status: OrderStatus
    fills: list[Fill]

    @property
    def total_filled(self) -> int:
        return sum(f.count for f in self.fills)


class SafetyRailViolation(RuntimeError):
    """Raised instead of submitting an order that fails a safety check
    (max order size, max daily loss, sanity check on API response, etc).
    Never silently downsize or skip -- the caller must see this and log it."""


class ExecutionEngine(ABC):
    @abstractmethod
    def submit_order(self, order: Order) -> OrderResult:
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, client_order_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_open_orders(self) -> list[Order]:
        raise NotImplementedError

    @abstractmethod
    def current_positions(self) -> dict[str, int]:
        """market_ticker -> signed contract count (positive = long YES)."""
        raise NotImplementedError

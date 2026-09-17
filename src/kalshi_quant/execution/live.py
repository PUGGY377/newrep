"""Live execution engine -- places real orders with real money.

Three independent things must all be true before this class will even
construct successfully:
  1. config.execution.mode == "live"                         (config.py)
  2. env var KALSHI_LIVE_TRADING_CONFIRMED == "true"          (config.py)
  3. the caller passes confirmed=True explicitly to __init__   (this file)

(1) and (2) are checked together by config.is_live_trading_authorized;
(3) exists so that even a script which somehow got a live-authorized
config can't instantiate this engine without a line of code that says,
in plain sight, "yes, really, place real orders." There is deliberately
no code path that can reach live order submission through only a config
file edit -- a human has to also flip the env var AND the calling script
has to also pass confirmed=True.

This is intentionally built and wired up LAST, after paper trading, per
the spec's build order.
"""

from __future__ import annotations

from datetime import datetime

from kalshi_quant.config import is_live_trading_authorized
from kalshi_quant.data.kalshi_client import KalshiClient, utc_now
from kalshi_quant.execution.audit import AuditLogger
from kalshi_quant.execution.base import (
    ExecutionEngine,
    Fill,
    Order,
    OrderAction,
    OrderResult,
    OrderStatus,
    SafetyRailViolation,
)


class LiveTradingNotAuthorizedError(RuntimeError):
    pass


class LiveExecutionEngine(ExecutionEngine):
    def __init__(
        self, kalshi_client: KalshiClient, audit_logger: AuditLogger, config: dict,
        max_single_order_usd: float, max_daily_loss_usd: float, confirmed: bool,
    ):
        if not confirmed:
            raise LiveTradingNotAuthorizedError(
                "LiveExecutionEngine requires confirmed=True to be passed explicitly by the "
                "calling script -- this is not something that can be set via config or env alone."
            )
        if not is_live_trading_authorized(config):
            raise LiveTradingNotAuthorizedError(
                "Live trading is not authorized: requires config.execution.mode == 'live' AND "
                "env var KALSHI_LIVE_TRADING_CONFIRMED == 'true'. Refusing to place real orders."
            )
        if kalshi_client.environment != "prod":
            raise LiveTradingNotAuthorizedError(
                f"KalshiClient is configured for environment={kalshi_client.environment!r}, "
                "not 'prod'. Refusing to place live orders against a non-prod environment."
            )

        self._client = kalshi_client
        self._audit = audit_logger
        self._max_single_order_usd = max_single_order_usd
        self._max_daily_loss_usd = max_daily_loss_usd
        self._realized_pnl_today = 0.0
        self._current_day = utc_now().date().isoformat()

    def _roll_day_if_needed(self) -> None:
        today = utc_now().date().isoformat()
        if self._current_day != today:
            self._current_day = today
            self._realized_pnl_today = 0.0

    def _check_safety_rails(self, order: Order) -> None:
        self._roll_day_if_needed()
        notional = order.count * order.limit_price_cents / 100.0
        if notional > self._max_single_order_usd:
            raise SafetyRailViolation(
                f"Order notional ${notional:.2f} exceeds max_single_order_usd "
                f"${self._max_single_order_usd:.2f}"
            )
        if self._realized_pnl_today <= -self._max_daily_loss_usd:
            raise SafetyRailViolation(
                f"Daily loss limit reached; no new live positions today."
            )

    def submit_order(self, order: Order) -> OrderResult:
        now = utc_now()
        self._check_safety_rails(order)  # let SafetyRailViolation propagate -- do NOT silently downsize
        self._audit.log_order_submitted(now, order)

        payload = {
            "ticker": order.market_ticker,
            "client_order_id": order.client_order_id,
            "side": order.side.value,
            "action": order.action.value,
            "count": order.count,
            "type": "limit",
            f"{order.side.value}_price": order.limit_price_cents,
        }
        response = self._client.create_order(payload)

        # Sanity-check the response before treating it as truth -- never
        # assume an order was placed as requested just because the HTTP
        # call didn't raise.
        resp_order = response.get("order", response)
        if resp_order.get("ticker") != order.market_ticker:
            raise SafetyRailViolation(
                f"Kalshi order response ticker {resp_order.get('ticker')!r} does not match "
                f"requested {order.market_ticker!r}; refusing to trust this response."
            )

        self._audit.log(
            "live_order_response", now, order=order, raw_response=response,
        )

        status_str = resp_order.get("status", "pending")
        status = {
            "resting": OrderStatus.RESTING, "filled": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELED, "pending": OrderStatus.PENDING,
        }.get(status_str, OrderStatus.PENDING)

        fills = []
        for raw_fill in response.get("fills", []):
            fill = Fill(
                client_order_id=order.client_order_id, market_ticker=order.market_ticker,
                side=order.side, count=int(raw_fill.get("count", 0)),
                price_cents=int(raw_fill.get("price", 0)), filled_at=now,
            )
            fills.append(fill)
            self._audit.log_fill(now, fill)

        return OrderResult(order=order, status=status, fills=fills)

    def cancel_order(self, client_order_id: str) -> None:
        self._client.cancel_order(client_order_id)

    def get_open_orders(self) -> list[Order]:
        raise NotImplementedError(
            "Mapping Kalshi's raw order response back into Order objects needs the exact "
            "response schema confirmed against a live/demo account before being trusted -- "
            "not implemented yet. Use KalshiClient.get_orders() directly if you need raw data."
        )

    def current_positions(self) -> dict[str, int]:
        positions = {}
        for raw in self._client.get_positions():
            ticker = raw.get("ticker")
            if ticker:
                positions[ticker] = int(raw.get("position", 0))
        return positions

"""Paper trading execution engine -- the default execution mode.

Simulates fills against REAL live order book data pulled from Kalshi
(read-only endpoints, no auth needed for market data beyond what the
client already requires) so paper P&L reflects real market conditions,
while never sending an order that could execute for real money.

Documented simplification (v1): orders fill immediately as marketable
limit orders -- as much as the book supports at or better than the limit
price, right now -- rather than resting on the book and being simulated
across future snapshots with requoting. `cancel_order` and
`get_open_orders` are therefore trivial in this version (nothing rests).
Modeling resting-order dynamics (queue position, partial fills over time,
requoting on adverse moves) is real future work flagged here rather than
silently assumed away -- see the module docstring's "immediate-or-cancel"
framing repeated in `submit_order`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from kalshi_quant.data.kalshi_client import KalshiClient, utc_now
from kalshi_quant.data.models import OrderbookLevel, OrderbookSnapshot, Side
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


@dataclass
class PaperAccountState:
    starting_capital: float
    cash: float
    positions: dict[str, dict[str, int]] = field(default_factory=dict)  # ticker -> {"yes": n, "no": n}
    cost_basis_dollars: dict[str, dict[str, float]] = field(default_factory=dict)  # ticker -> side -> $ spent
    realized_pnl_today: float = 0.0
    current_day: str = ""

    def position_count(self, ticker: str, side: Side) -> int:
        return self.positions.get(ticker, {}).get(side.value, 0)

    def adjust_position(self, ticker: str, side: Side, delta: int, cost_delta_dollars: float = 0.0) -> None:
        self.positions.setdefault(ticker, {"yes": 0, "no": 0})
        self.positions[ticker][side.value] += delta
        self.cost_basis_dollars.setdefault(ticker, {"yes": 0.0, "no": 0.0})
        self.cost_basis_dollars[ticker][side.value] += cost_delta_dollars


def _snapshot_from_raw(ticker: str, raw: dict) -> OrderbookSnapshot:
    """Adapts KalshiClient.get_orderbook's raw JSON into our OrderbookSnapshot.

    Kalshi's v2 orderbook response carries exactly two resting-BID arrays,
    `yes` and `no` (each a list of [price_cents, count] levels) -- there is
    no separate "ask" array, because a YES ask at price P is economically
    identical to a resting NO bid at (100 - P) (buying YES at P == selling
    NO at 100-P), and vice versa. So the ask side of each book is DERIVED
    from the other side's bids, not read directly. This derivation is
    documented Kalshi behavior, not a guess -- but as with the rest of the
    client, re-verify field names (`yes`/`no`) against a live response if
    this ever throws a KeyError, since Kalshi has adjusted response
    shapes before.
    """
    def parse_levels(levels: list) -> list[tuple[int, int]]:
        return [(int(p), int(c)) for p, c in levels]

    book = raw.get("orderbook", raw)
    yes_bid_levels = sorted(parse_levels(book.get("yes", [])), key=lambda x: -x[0])
    no_bid_levels = sorted(parse_levels(book.get("no", [])), key=lambda x: -x[0])

    # no_bid_levels/yes_bid_levels are sorted best-bid-first (descending
    # price), so 100-price is ascending -- i.e. already cheapest-ask-first.
    yes_asks = tuple(OrderbookLevel(price_cents=100 - p, contracts=c) for p, c in no_bid_levels)
    no_asks = tuple(OrderbookLevel(price_cents=100 - p, contracts=c) for p, c in yes_bid_levels)

    return OrderbookSnapshot(
        market_ticker=ticker, as_of=utc_now(),
        yes_bids=tuple(OrderbookLevel(price_cents=p, contracts=c) for p, c in yes_bid_levels),
        yes_asks=yes_asks,
        no_bids=tuple(OrderbookLevel(price_cents=p, contracts=c) for p, c in no_bid_levels),
        no_asks=no_asks,
    )


class PaperExecutionEngine(ExecutionEngine):
    def __init__(
        self, kalshi_client: KalshiClient, audit_logger: AuditLogger, starting_capital: float,
        max_single_order_usd: float, max_daily_loss_usd: float,
    ):
        self._client = kalshi_client
        self._audit = audit_logger
        self._max_single_order_usd = max_single_order_usd
        self._max_daily_loss_usd = max_daily_loss_usd
        self.state = PaperAccountState(starting_capital=starting_capital, cash=starting_capital)
        self._roll_day_if_needed()

    def _roll_day_if_needed(self) -> None:
        today = utc_now().date().isoformat()
        if self.state.current_day != today:
            self.state.current_day = today
            self.state.realized_pnl_today = 0.0

    def _check_safety_rails(self, order: Order) -> None:
        self._roll_day_if_needed()
        notional = order.count * order.limit_price_cents / 100.0
        if notional > self._max_single_order_usd:
            raise SafetyRailViolation(
                f"Order notional ${notional:.2f} exceeds max_single_order_usd "
                f"${self._max_single_order_usd:.2f}"
            )
        if self.state.realized_pnl_today <= -self._max_daily_loss_usd:
            raise SafetyRailViolation(
                f"Daily loss limit reached (${self.state.realized_pnl_today:.2f} <= "
                f"-${self._max_daily_loss_usd:.2f}); no new positions today."
            )

    def submit_order(self, order: Order) -> OrderResult:
        now = utc_now()
        try:
            self._check_safety_rails(order)
        except SafetyRailViolation as e:
            self._audit.log_safety_rail_block(now, str(e), {"order": order})
            return OrderResult(order=order, status=OrderStatus.REJECTED, fills=[])

        self._audit.log_order_submitted(now, order)

        raw_book = self._client.get_orderbook(order.market_ticker)
        book = _snapshot_from_raw(order.market_ticker, raw_book)
        levels = book.yes_asks if order.side == Side.YES else book.no_asks

        remaining = order.count
        fills: list[Fill] = []
        for level in levels:
            if remaining <= 0:
                break
            if order.action == OrderAction.BUY and level.price_cents > order.limit_price_cents:
                break  # book sorted best-first; nothing better remains
            take = min(remaining, level.contracts)
            if take <= 0:
                continue
            fill = Fill(
                client_order_id=order.client_order_id, market_ticker=order.market_ticker,
                side=order.side, count=take, price_cents=level.price_cents, filled_at=now,
            )
            fills.append(fill)
            self._audit.log_fill(now, fill)
            cost = take * level.price_cents / 100.0
            self.state.adjust_position(order.market_ticker, order.side, take, cost_delta_dollars=cost)
            self.state.cash -= cost
            remaining -= take

        total_filled = sum(f.count for f in fills)
        if total_filled == 0:
            status = OrderStatus.REJECTED
        elif total_filled < order.count:
            status = OrderStatus.PARTIALLY_FILLED
        else:
            status = OrderStatus.FILLED

        return OrderResult(order=order, status=status, fills=fills)

    def cancel_order(self, client_order_id: str) -> None:
        return  # nothing rests in this version -- see module docstring

    def get_open_orders(self) -> list[Order]:
        return []

    def current_positions(self) -> dict[str, int]:
        """Net signed position per ticker: +n for YES contracts, -n for NO
        contracts (both simultaneously nonzero would mean an internal bug
        in sizing letting us hold both sides of the same market)."""
        result = {}
        for ticker, sides in self.state.positions.items():
            result[ticker] = sides.get("yes", 0) - sides.get("no", 0)
        return result

    def settle_position(self, ticker: str, side: Side, won: bool, when: datetime) -> None:
        """Called by the paper-trading loop once a market's real Kalshi
        settlement result is observed. Settles the ENTIRE currently-held
        position on `side` for this market at once (matching Kalshi's own
        settlement semantics -- a market resolves fully, not partially),
        realizing P&L into cash and the daily-loss tracker the safety
        rails consult."""
        self._roll_day_if_needed()
        contracts = self.state.position_count(ticker, side)
        if contracts == 0:
            return
        cost_basis = self.state.cost_basis_dollars.get(ticker, {}).get(side.value, 0.0)
        payoff = float(contracts) if won else 0.0
        pnl = payoff - cost_basis

        self.state.cash += payoff
        self.state.adjust_position(ticker, side, -contracts, cost_delta_dollars=-cost_basis)
        self.state.realized_pnl_today += pnl
        self._audit.log("settlement", when, market_ticker=ticker, side=side, contracts=contracts,
                         won=won, payoff=payoff, cost_basis=cost_basis, pnl=pnl)

from datetime import datetime

import pytest

from kalshi_quant.data.models import Side
from kalshi_quant.execution.audit import AuditLogger
from kalshi_quant.execution.base import Order, OrderAction, OrderStatus, SafetyRailViolation
from kalshi_quant.execution.paper import PaperExecutionEngine


class FakeKalshiClient:
    environment = "demo"

    def __init__(self, orderbook: dict):
        self._orderbook = orderbook

    def get_orderbook(self, ticker: str) -> dict:
        return self._orderbook


def make_engine(tmp_path, orderbook, starting_capital=500.0, max_single_order_usd=100.0,
                 max_daily_loss_usd=50.0) -> PaperExecutionEngine:
    client = FakeKalshiClient(orderbook)
    audit = AuditLogger(log_dir=tmp_path / "audit")
    return PaperExecutionEngine(client, audit, starting_capital, max_single_order_usd, max_daily_loss_usd)


def make_order(ticker="KXTEST-A", side=Side.YES, count=10, limit_price_cents=60) -> Order:
    return Order(
        client_order_id="order-1", market_ticker=ticker, side=side, action=OrderAction.BUY,
        count=count, limit_price_cents=limit_price_cents, created_at=datetime(2024, 1, 1),
    )


def test_submit_order_fills_from_derived_ask_side(tmp_path):
    # yes ask = 100 - best no bid; no bid of 40 -> yes ask of 60
    orderbook = {"orderbook": {"yes": [], "no": [[40, 20]]}}
    engine = make_engine(tmp_path, orderbook)
    order = make_order(count=10, limit_price_cents=60)
    result = engine.submit_order(order)

    assert result.status == OrderStatus.FILLED
    assert result.total_filled == 10
    assert engine.current_positions()["KXTEST-A"] == 10
    assert engine.state.cash == pytest.approx(500.0 - 10 * 0.60)


def test_submit_order_partial_fill_when_book_thin(tmp_path):
    orderbook = {"orderbook": {"yes": [], "no": [[40, 5]]}}
    engine = make_engine(tmp_path, orderbook)
    order = make_order(count=10, limit_price_cents=60)
    result = engine.submit_order(order)
    assert result.status == OrderStatus.PARTIALLY_FILLED
    assert result.total_filled == 5


def test_submit_order_rejects_when_price_above_limit(tmp_path):
    orderbook = {"orderbook": {"yes": [], "no": [[30, 20]]}}  # implies yes ask of 70
    engine = make_engine(tmp_path, orderbook)
    order = make_order(count=10, limit_price_cents=60)  # limit too low to cross
    result = engine.submit_order(order)
    assert result.status == OrderStatus.REJECTED
    assert result.total_filled == 0


def test_safety_rail_blocks_oversized_order(tmp_path):
    orderbook = {"orderbook": {"yes": [], "no": [[40, 1000]]}}
    engine = make_engine(tmp_path, orderbook, max_single_order_usd=10.0)
    order = make_order(count=100, limit_price_cents=60)  # notional $60 > $10 cap
    result = engine.submit_order(order)
    assert result.status == OrderStatus.REJECTED
    assert result.total_filled == 0
    assert engine.current_positions().get("KXTEST-A", 0) == 0


def test_daily_loss_limit_blocks_new_orders(tmp_path):
    orderbook = {"orderbook": {"yes": [], "no": [[40, 1000]]}}
    engine = make_engine(tmp_path, orderbook, max_daily_loss_usd=5.0)
    engine.state.realized_pnl_today = -10.0  # already breached
    order = make_order(count=1, limit_price_cents=60)
    result = engine.submit_order(order)
    assert result.status == OrderStatus.REJECTED


def test_settle_position_realizes_pnl_correctly(tmp_path):
    orderbook = {"orderbook": {"yes": [], "no": [[40, 20]]}}  # yes ask = 60
    engine = make_engine(tmp_path, orderbook)
    order = make_order(count=10, limit_price_cents=60)
    engine.submit_order(order)
    cash_after_fill = engine.state.cash

    engine.settle_position("KXTEST-A", Side.YES, won=True, when=datetime(2024, 1, 2))
    assert engine.state.cash == pytest.approx(cash_after_fill + 10.0)  # payoff = $1/contract
    assert engine.state.realized_pnl_today == pytest.approx(10.0 - 10 * 0.60)
    assert engine.current_positions().get("KXTEST-A", 0) == 0


def test_settle_position_losing_trade(tmp_path):
    orderbook = {"orderbook": {"yes": [], "no": [[40, 20]]}}
    engine = make_engine(tmp_path, orderbook)
    engine.submit_order(make_order(count=10, limit_price_cents=60))
    engine.settle_position("KXTEST-A", Side.YES, won=False, when=datetime(2024, 1, 2))
    assert engine.state.realized_pnl_today == pytest.approx(-10 * 0.60)


def test_audit_log_written_to_disk(tmp_path):
    orderbook = {"orderbook": {"yes": [], "no": [[40, 20]]}}
    engine = make_engine(tmp_path, orderbook)
    engine.submit_order(make_order())
    log_files = list((tmp_path / "audit").glob("*.jsonl"))
    assert len(log_files) == 1
    content = log_files[0].read_text()
    assert "order_submitted" in content
    assert "fill" in content

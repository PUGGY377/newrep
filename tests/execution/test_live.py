from datetime import datetime

import pytest

from kalshi_quant.data.models import Side
from kalshi_quant.execution.audit import AuditLogger
from kalshi_quant.execution.base import Order, OrderAction
from kalshi_quant.execution.live import LiveExecutionEngine, LiveTradingNotAuthorizedError


class FakeKalshiClient:
    def __init__(self, environment="prod"):
        self.environment = environment
        self.orders_created = []

    def create_order(self, order):
        self.orders_created.append(order)
        return {"order": {"ticker": order["ticker"], "status": "filled"}, "fills": []}


def make_order():
    return Order(
        client_order_id="o1", market_ticker="KXTEST-A", side=Side.YES, action=OrderAction.BUY,
        count=1, limit_price_cents=50, created_at=datetime(2024, 1, 1),
    )


def authorized_config():
    return {"execution": {"mode": "live"}}


def test_requires_confirmed_flag(tmp_path, monkeypatch):
    monkeypatch.setenv("KALSHI_LIVE_TRADING_CONFIRMED", "true")
    with pytest.raises(LiveTradingNotAuthorizedError):
        LiveExecutionEngine(
            FakeKalshiClient(), AuditLogger(tmp_path), authorized_config(), 100.0, 50.0, confirmed=False,
        )


def test_requires_env_var_even_if_confirmed_true(tmp_path, monkeypatch):
    monkeypatch.setenv("KALSHI_LIVE_TRADING_CONFIRMED", "false")
    with pytest.raises(LiveTradingNotAuthorizedError):
        LiveExecutionEngine(
            FakeKalshiClient(), AuditLogger(tmp_path), authorized_config(), 100.0, 50.0, confirmed=True,
        )


def test_requires_config_mode_live(tmp_path, monkeypatch):
    monkeypatch.setenv("KALSHI_LIVE_TRADING_CONFIRMED", "true")
    with pytest.raises(LiveTradingNotAuthorizedError):
        LiveExecutionEngine(
            FakeKalshiClient(), AuditLogger(tmp_path), {"execution": {"mode": "paper"}}, 100.0, 50.0,
            confirmed=True,
        )


def test_requires_prod_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("KALSHI_LIVE_TRADING_CONFIRMED", "true")
    with pytest.raises(LiveTradingNotAuthorizedError):
        LiveExecutionEngine(
            FakeKalshiClient(environment="demo"), AuditLogger(tmp_path), authorized_config(), 100.0, 50.0,
            confirmed=True,
        )


def test_all_gates_pass_allows_construction_and_submits(tmp_path, monkeypatch):
    monkeypatch.setenv("KALSHI_LIVE_TRADING_CONFIRMED", "true")
    client = FakeKalshiClient(environment="prod")
    engine = LiveExecutionEngine(
        client, AuditLogger(tmp_path), authorized_config(), 100.0, 50.0, confirmed=True,
    )
    result = engine.submit_order(make_order())
    assert len(client.orders_created) == 1
    assert result.status.value == "filled"


def test_safety_rail_blocks_oversized_live_order(tmp_path, monkeypatch):
    monkeypatch.setenv("KALSHI_LIVE_TRADING_CONFIRMED", "true")
    client = FakeKalshiClient(environment="prod")
    engine = LiveExecutionEngine(
        client, AuditLogger(tmp_path), authorized_config(), max_single_order_usd=0.10,
        max_daily_loss_usd=50.0, confirmed=True,
    )
    from kalshi_quant.execution.base import SafetyRailViolation
    with pytest.raises(SafetyRailViolation):
        engine.submit_order(make_order())
    assert len(client.orders_created) == 0

from datetime import datetime, timezone

import pytest

from kalshi_quant.data.models import OrderbookLevel, OrderbookSnapshot


def make_orderbook(
    ticker: str = "KXTEST-24-A",
    yes_bids: list[tuple[int, int]] | None = None,
    yes_asks: list[tuple[int, int]] | None = None,
    no_bids: list[tuple[int, int]] | None = None,
    no_asks: list[tuple[int, int]] | None = None,
    as_of: datetime | None = None,
) -> OrderbookSnapshot:
    def levels(pairs: list[tuple[int, int]] | None) -> tuple[OrderbookLevel, ...]:
        return tuple(OrderbookLevel(price_cents=p, contracts=c) for p, c in (pairs or []))

    return OrderbookSnapshot(
        market_ticker=ticker,
        as_of=as_of or datetime(2024, 1, 1, tzinfo=timezone.utc),
        yes_bids=levels(yes_bids if yes_bids is not None else [(54, 100)]),
        yes_asks=levels(yes_asks if yes_asks is not None else [(56, 100)]),
        no_bids=levels(no_bids if no_bids is not None else [(44, 100)]),
        no_asks=levels(no_asks if no_asks is not None else [(46, 100)]),
    )


@pytest.fixture
def orderbook_factory():
    return make_orderbook

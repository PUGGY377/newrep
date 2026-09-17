import pytest

from kalshi_quant.data.models import Side
from kalshi_quant.risk.liquidity import estimate_fill, max_contracts_within_slippage
from tests.conftest import make_orderbook


def test_estimate_fill_single_level():
    ob = make_orderbook(yes_asks=[(50, 100)])
    fill = estimate_fill(ob, Side.YES, 40)
    assert fill.contracts_fillable == 40
    assert fill.average_price_probability == pytest.approx(0.50)
    assert fill.slippage_vs_best_price == pytest.approx(0.0)


def test_estimate_fill_walks_multiple_levels():
    ob = make_orderbook(yes_asks=[(50, 10), (55, 10), (60, 100)])
    fill = estimate_fill(ob, Side.YES, 25)
    expected_avg = (10 * 0.50 + 10 * 0.55 + 5 * 0.60) / 25
    assert fill.contracts_fillable == 25
    assert fill.average_price_probability == pytest.approx(expected_avg)
    assert fill.slippage_vs_best_price == pytest.approx(expected_avg - 0.50)


def test_estimate_fill_partial_when_book_thin():
    ob = make_orderbook(yes_asks=[(50, 5)])
    fill = estimate_fill(ob, Side.YES, 100)
    assert fill.contracts_fillable == 5


def test_estimate_fill_empty_book():
    ob = make_orderbook(yes_asks=[])
    fill = estimate_fill(ob, Side.YES, 10)
    assert fill.contracts_fillable == 0


def test_max_contracts_within_slippage_respects_depth_fraction():
    ob = make_orderbook(yes_asks=[(50, 100)])
    n = max_contracts_within_slippage(
        ob, Side.YES, max_acceptable_slippage=1.0, max_orderbook_depth_fraction=0.25,
    )
    assert n == 25


def test_max_contracts_within_slippage_respects_price_budget():
    ob = make_orderbook(yes_asks=[(50, 10), (99, 1000)])
    # allow full depth fraction, but slippage budget should stop us near the cheap level
    n = max_contracts_within_slippage(
        ob, Side.YES, max_acceptable_slippage=0.01, max_orderbook_depth_fraction=1.0,
    )
    fill = estimate_fill(ob, Side.YES, n)
    assert fill.slippage_vs_best_price <= 0.01 + 1e-9
    # one more contract should breach the budget
    fill_plus_one = estimate_fill(ob, Side.YES, n + 1)
    assert fill_plus_one.slippage_vs_best_price > 0.01

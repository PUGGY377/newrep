from datetime import datetime

import pytest

from kalshi_quant.signal.fees import FeeSchedule, current_fee_schedule, fee_schedule_as_of


def test_fee_formula_matches_documented_shape():
    schedule = FeeSchedule(
        version="test", effective_from=datetime(2020, 1, 1), effective_to=None,
        general_multiplier=0.07,
    )
    # price=0.5 maximizes p*(1-p): fee = ceil(0.07*10*0.5*0.5*100)/100
    fee = schedule.fee_dollars("KXTEST", contracts=10, price_probability=0.5)
    assert fee == pytest.approx(math_ceil_cents(0.07 * 10 * 0.5 * 0.5))


def math_ceil_cents(x: float) -> float:
    import math
    return math.ceil(x * 100) / 100.0


def test_fee_is_zero_at_price_zero_or_one():
    schedule = FeeSchedule(
        version="test", effective_from=datetime(2020, 1, 1), effective_to=None,
        general_multiplier=0.07,
    )
    assert schedule.fee_dollars("KXTEST", 10, 0.0) == 0.0
    assert schedule.fee_dollars("KXTEST", 10, 1.0) == 0.0


def test_reduced_multiplier_applies_only_to_matching_prefix():
    schedule = FeeSchedule(
        version="test", effective_from=datetime(2020, 1, 1), effective_to=None,
        general_multiplier=0.07, reduced_multiplier=0.035, reduced_series_prefixes=("KXSPX",),
    )
    reduced = schedule.fee_dollars("KXSPXRANGE", 10, 0.5)
    general = schedule.fee_dollars("KXOTHER", 10, 0.5)
    assert reduced == pytest.approx(general / 2)


def test_invalid_price_raises():
    schedule = current_fee_schedule()
    with pytest.raises(ValueError):
        schedule.fee_dollars("KXTEST", 10, 1.5)


def test_fee_schedule_as_of_selects_correct_version():
    old = FeeSchedule(
        version="old", effective_from=datetime(2020, 1, 1), effective_to=datetime(2023, 1, 1),
        general_multiplier=0.05,
    )
    new = FeeSchedule(
        version="new", effective_from=datetime(2023, 1, 1), effective_to=None,
        general_multiplier=0.07,
    )
    import kalshi_quant.signal.fees as fees_module
    original = fees_module.FEE_SCHEDULES
    fees_module.FEE_SCHEDULES = [old, new]
    try:
        assert fee_schedule_as_of(datetime(2021, 6, 1)).version == "old"
        assert fee_schedule_as_of(datetime(2024, 6, 1)).version == "new"
        with pytest.raises(ValueError):
            fee_schedule_as_of(datetime(2019, 1, 1))
    finally:
        fees_module.FEE_SCHEDULES = original
